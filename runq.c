/* Inference for Llama-2 Transformer model in pure C, int8 quantized forward pass. */

#include <stdio.h> // Ensure stdio.h is included for file operations
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string.h>
// #include <fcntl.h> // Removed: No longer needed for mmap
#if defined _WIN32
    #include "win.h" // For Windows-specific timing or other functions
#else
    #include <unistd.h> // For POSIX functions like close (if still needed elsewhere), ssize_t
    // #include <sys/mman.h> // Removed: mmap is no longer used
#endif
// ----------------------------------------------------------------------------
// Globals
int GS = 0; // group size global for quantization of the weights

// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    int8_t* q;     // quantized values
    float* s;   // scaling factors
} QuantizedTensor;

typedef struct {
    // token embedding table
    QuantizedTensor *q_tokens; // (vocab_size, dim)
    float* token_embedding_table; // same, but dequantized

    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    QuantizedTensor *wq; // (layer, dim, n_heads * head_size)
    QuantizedTensor *wk; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wv; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    QuantizedTensor *w1; // (layer, hidden_dim, dim)
    QuantizedTensor *w2; // (layer, dim, hidden_dim)
    QuantizedTensor *w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    QuantizedTensor *wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    QuantizedTensor xq; // quantized x (dim,)
    QuantizedTensor hq; // quantized hb (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // State for managing file data if not using mmap
    // int fd; // Removed: No longer using mmap's file descriptor
    float* data; // pointer to memory holding checkpoint data (either mmap'ed or malloc'ed)
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->xq = (QuantizedTensor) { .q = calloc(p->dim, sizeof(int8_t)), .s = calloc(p->dim / GS, sizeof(float)) }; // Corrected scale size
    s->hq = (QuantizedTensor) { .q = calloc(p->hidden_dim, sizeof(int8_t)), .s = calloc(p->hidden_dim / GS, sizeof(float)) }; // Corrected scale size
    s->q = calloc(p->dim, sizeof(float));
    s->k = calloc(kv_dim, sizeof(float));
    s->v = calloc(kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->xq.q || !s->xq.s || !s->hq.q || !s->hq.s
     || !s->q || !s->k || !s->v || !s->att || !s->logits || !s->key_cache
     || !s->value_cache) {
        fprintf(stderr, "malloc failed in malloc_run_state!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->xq.q);
    free(s->xq.s);
    free(s->hq.q);
    free(s->hq.s);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

// ----------------------------------------------------------------------------
// Quantization functions

void dequantize(QuantizedTensor *qx, float* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = (float)qx->q[i] * qx->s[i / GS];
    }
}

void quantize(QuantizedTensor *qx, float* x, int n) {
    int num_groups = n / GS;
    float Q_MAX = 127.0f;

    for (int group = 0; group < num_groups; group++) {
        // find the max absolute value in the current group
        float wmax = 0.0;
        for (int i = 0; i < GS; i++) {
            float val = fabsf(x[group * GS + i]); // Use fabsf for float
            if (val > wmax) {
                wmax = val;
            }
        }

        // calculate and write the scaling factor
        float scale = wmax / Q_MAX;
        if (scale == 0.0f) scale = 1.0f / Q_MAX; // Avoid division by zero if wmax is 0
        qx->s[group] = scale;

        // calculate and write the quantized values
        for (int i = 0; i < GS; i++) {
            float quant_value = x[group * GS + i] / scale; // scale
            int8_t quantized = (int8_t) roundf(quant_value); // round and clamp, use roundf for float
            qx->q[group * GS + i] = quantized;
        }
    }
}

/* initialize `n` x quantized tensor (with `size_each` elements), starting from memory pointed at *ptr */
QuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each) {
    void *p = *ptr;
    QuantizedTensor *res = malloc(n * sizeof(QuantizedTensor));
    if (!res) {
        fprintf(stderr, "malloc failed in init_quantized_tensors for res array!\n");
        exit(EXIT_FAILURE);
    }
    for(int i=0; i<n; i++) {
        /* map quantized int8 values*/
        res[i].q = (int8_t*)p;
        p = (int8_t*)p + size_each;
        /* map scale factors */
        res[i].s = (float*)p;
        p = (float*)p + size_each / GS;
    }
    *ptr = p; // advance ptr to current position
    return res;
}

void memory_map_weights(TransformerWeights *w, Config* p, void* ptr, uint8_t shared_classifier) {
    int head_size = p->dim / p->n_heads;
    // first are the parameters that are kept in fp32 (the rmsnorm (1D) weights)
    float* fptr = (float*) ptr; // cast our pointer to float*
    w->rms_att_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_ffn_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_final_weight = fptr;
    fptr += p->dim;

    // now read all the quantized weights
    ptr = (void*)fptr; // now cast the pointer back to void*
    w->q_tokens = init_quantized_tensors(&ptr, 1, p->vocab_size * p->dim);
    // dequantize token embedding table
    w->token_embedding_table = malloc(p->vocab_size * p->dim * sizeof(float));
    if (!w->token_embedding_table) {
        fprintf(stderr, "malloc failed for token_embedding_table!\n");
        exit(EXIT_FAILURE);
    }
    dequantize(w->q_tokens, w->token_embedding_table, p->vocab_size * p->dim);

    w->wq = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_heads * head_size));
    w->wk = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
    w->wv = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
    w->wo = init_quantized_tensors(&ptr, p->n_layers, (p->n_heads * head_size) * p->dim);

    w->w1 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);
    w->w2 = init_quantized_tensors(&ptr, p->n_layers, p->hidden_dim * p->dim);
    w->w3 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);

    w->wcls = shared_classifier ? w->q_tokens : init_quantized_tensors(&ptr, 1, p->dim * p->vocab_size);
}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     float** data, ssize_t* file_size) { // fd parameter removed
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    
    // Read magic number
    uint32_t magic_number;
    if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1) { 
        fprintf(stderr, "Failed to read magic number\n"); fclose(file); exit(EXIT_FAILURE); 
    }
    if (magic_number != 0x616b3432) { 
        fprintf(stderr, "Bad magic number: 0x%x\n", magic_number); fclose(file); exit(EXIT_FAILURE); 
    }

    // Read version
    int version;
    if (fread(&version, sizeof(int), 1, file) != 1) { 
        fprintf(stderr, "Failed to read version\n"); fclose(file); exit(EXIT_FAILURE); 
    }
    if (version != 2) { 
        fprintf(stderr, "Bad version %d, need version 2\n", version); fclose(file); exit(EXIT_FAILURE); 
    }
    
    int header_size = 256; // The header size for version 2 in bytes (fixed)

    // Read Config
    if (fread(config, sizeof(Config), 1, file) != 1) { 
        fprintf(stderr, "Failed to read config\n"); fclose(file); exit(EXIT_FAILURE); 
    }
    
    // Read flags
    uint8_t shared_classifier;
    if (fread(&shared_classifier, sizeof(uint8_t), 1, file) != 1) { 
        fprintf(stderr, "Failed to read shared_classifier flag\n"); fclose(file); exit(EXIT_FAILURE); 
    }
    int group_size_from_file; // Renamed to avoid conflict with global GS if any
    if (fread(&group_size_from_file, sizeof(int), 1, file) != 1) { 
        fprintf(stderr, "Failed to read group_size\n"); fclose(file); exit(EXIT_FAILURE); 
    }
    GS = group_size_from_file; // Set global GS

    // Figure out the file size
    if (fseek(file, 0, SEEK_END) != 0) {
        fprintf(stderr, "fseek to end of file failed\n"); fclose(file); exit(EXIT_FAILURE);
    }
    *file_size = ftell(file);
    if (*file_size == -1) {
        fprintf(stderr, "ftell failed to get file size\n"); fclose(file); exit(EXIT_FAILURE);
    }
    
    // Allocate memory for the entire file content
    *data = malloc(*file_size);
    if (*data == NULL) {
        fprintf(stderr, "malloc failed to allocate %zd bytes for checkpoint!\n", *file_size);
        fclose(file);
        exit(EXIT_FAILURE);
    }

    // Rewind to the beginning of the file to read its content
    if (fseek(file, 0, SEEK_SET) != 0) {
        fprintf(stderr, "fseek to beginning of file failed\n");
        free(*data); // Free allocated memory
        fclose(file);
        exit(EXIT_FAILURE);
    }
    
    // Read the entire file content into the allocated buffer
    if (fread(*data, *file_size, 1, file) != 1) {
        fprintf(stderr, "fread failed to read checkpoint into memory (read %zd bytes)\n", *file_size);
        free(*data); // Free allocated memory
        fclose(file);
        exit(EXIT_FAILURE);
    }
    fclose(file); // Close the file as its content is now in memory

    // Weights pointer starts after the header in the buffer
    void* weights_ptr = ((char*)*data) + header_size;
    memory_map_weights(weights, config, weights_ptr, shared_classifier);
}


void build_transformer(Transformer *t, char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    // Free QuantizedTensors (pointers within TransformerWeights that were malloced by init_quantized_tensors)
    // Note: The .q and .s pointers within QuantizedTensor structs point into the main 't->data' block
    // or into t->weights.token_embedding_table.
    // init_quantized_tensors itself mallocs the array of QuantizedTensor structs.
    
    if (t->weights.q_tokens) free(t->weights.q_tokens); // This is the array of QuantizedTensor structs
    if (t->weights.token_embedding_table) free(t->weights.token_embedding_table); // This was separately malloced

    if (t->weights.wq) free(t->weights.wq);
    if (t->weights.wk) free(t->weights.wk);
    if (t->weights.wv) free(t->weights.wv);
    if (t->weights.wo) free(t->weights.wo);
    if (t->weights.w1) free(t->weights.w1);
    if (t->weights.w2) free(t->weights.w2);
    if (t->weights.w3) free(t->weights.w3);
    
    // wcls can be an alias to q_tokens or a separate allocation
    // The original code checks if wcls is different from q_tokens before freeing.
    // However, q_tokens is an array of 1 QuantizedTensor, wcls is also an array of 1.
    // The check should be on the .q pointers if we want to be precise about aliasing actual data.
    // But since init_quantized_tensors always mallocs a new QuantizedTensor array,
    // we should free wcls if it wasn't assigned to q_tokens (the array pointer itself).
    if (t->weights.wcls && t->weights.wcls != t->weights.q_tokens) {
         free(t->weights.wcls);
    }


    // Free the main data buffer that held the file contents
    if (t->data != NULL) {
        free(t->data);
        t->data = NULL;
    }
    
    // No fd to close as mmap is not used.
    // if (t->fd != -1) { close(t->fd); } // Removed

    // free the RunState buffers
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    // inputs to this function are both quantized

    // Log the parameters n and d to log.txt
    FILE *log_file = fopen("log.txt", "a"); // Open in append mode
    if (log_file != NULL) {
        fprintf(log_file, "matmul called with n=%d, d=%d\n", n, d);
        fclose(log_file);
    } else {
        // Optional: Handle error if the file cannot be opened
        // fprintf(stderr, "Warning: Could not open log.txt for writing.\n");
    }

    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        int32_t ival = 0;
        int in = i * n; // Start index in w's quantized data for the i-th row

        // Process in groups of GS
        for (int j_group_start = 0; j_group_start < n; j_group_start += GS) {
            int32_t group_ival = 0; // Accumulator for the current group, reset for each group
            // Iterate within the group
            for (int k_offset = 0; k_offset < GS; k_offset++) {
                int current_j = j_group_start + k_offset;
                if (current_j < n) { // Boundary check for the last group
                     // x->q index is (j_group_start + k_offset)
                     // w->q index is (in + j_group_start + k_offset)
                    group_ival += ((int32_t) x->q[current_j]) * ((int32_t) w->q[in + current_j]);
                }
            }
            // Apply scaling factors for the group
            // x->s index is (j_group_start / GS)
            // w->s index is ((in + j_group_start) / GS)
            val += ((float) group_ival) * w->s[(in + j_group_start) / GS] * x->s[j_group_start / GS];
        }
        xout[i] = val;
    }
}


float* forward(Transformer* transformer, int token, int pos) {

    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    memcpy(x, w->token_embedding_table + token*dim, dim * sizeof(float));

    // forward all the layers
    for(int l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // qkv matmuls for this position
        quantize(&s->xq, s->xb, dim);
        matmul(s->q, &s->xq, w->wq + l, dim, dim);
        matmul(s->k, &s->xq, w->wk + l, dim, kv_dim);
        matmul(s->v, &s->xq, w->wv + l, dim, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i+=2) {
            int head_dim_idx = i % head_size; // current dimension index within the head
            float freq = 1.0f / powf(10000.0f, (float)head_dim_idx / head_size);
            float val = (float)pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v_idx = 0; v_idx < rotn; v_idx++) {
                float* vec = v_idx == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // save key,value at this time step (pos) to our kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        float* key_cache_row = s->key_cache + loff + pos * kv_dim;
        float* value_cache_row = s->value_cache + loff + pos * kv_dim;
        memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
        memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row));

        // multihead attention. iterate over all heads
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            float* q_head = s->q + h * head_size;
            // attention scores for this head
            float* att_head = s->att + h * p->seq_len;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k_cached = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q_head[i] * k_cached[i];
                }
                score /= sqrtf((float)head_size);
                // save the score to the attention buffer
                att_head[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att_head, pos + 1);

            // weighted sum of the values, store back into xb
            float* xb_head = s->xb + h * head_size; // output for this head
            memset(xb_head, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float* v_cached = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                float a = att_head[t];
                // accumulate the weighted value into xb_head
                for (int i = 0; i < head_size; i++) {
                    xb_head[i] += a * v_cached[i];
                }
            }
        }

        // final matmul to get the output of the attention
        quantize(&s->xq, s->xb, dim);
        matmul(s->xb2, &s->xq, w->wo + l, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        quantize(&s->xq, s->xb, dim);
        matmul(s->hb, &s->xq, w->w1 + l, dim, hidden_dim);
        matmul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        quantize(&s->hq, s->hb, hidden_dim);
        matmul(s->xb, &s->hq, w->w2 + l, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    quantize(&s->xq, x, dim);
    matmul(s->logits, &s->xq, w->wcls, dim, p->vocab_size);
    return s->logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    if(!t->vocab) {fprintf(stderr, "malloc failed for vocab strings\n"); exit(EXIT_FAILURE);}
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    if(!t->vocab_scores) {fprintf(stderr, "malloc failed for vocab_scores\n"); exit(EXIT_FAILURE);}

    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read max_token_length\n"); fclose(file); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read vocab_scores for token %d\n", i); fclose(file); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read len for token %d\n", i); fclose(file); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if(!t->vocab[i]) {fprintf(stderr, "malloc failed for vocab token %d\n", i); fclose(file); exit(EXIT_FAILURE);}
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read vocab string for token %d\n", i); fclose(file); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab); // sorted_vocab is malloced in encode if NULL
}

char* decode(Tokenizer* t, int prev_token, int token) {
    if (token < 0 || token >= t->vocab_size) {
        // fprintf(stderr, "Warning: token %d out of vocab range [0, %d)\n", token, t->vocab_size);
        return "?"; // Or some other indicator for invalid token
    }
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        // Ensure byte_val is within bounds for byte_pieces
        if (byte_val < 256) { // byte_val is unsigned char, so always >= 0
             piece = (char*)t->byte_pieces + byte_val * 2;
        } else {
            // This case should ideally not happen if vocab is correct
            // fprintf(stderr, "Warning: byte_val %u from token %s is out of range\n", byte_val, t->vocab[token]);
            return "?"; 
        }
    }
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        if(!t->sorted_vocab) {fprintf(stderr, "malloc failed for sorted_vocab\n"); exit(EXIT_FAILURE);}
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    if(!str_buffer) {fprintf(stderr, "malloc failed for str_buffer in encode\n"); exit(EXIT_FAILURE);}
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        if (dummy_prefix != -1) { // Make sure the space token exists
             tokens[(*n_tokens)++] = dummy_prefix;
        } else {
            // This is a fallback, ideally space token should exist.
            // Or handle this more gracefully depending on tokenizer's design.
            // fprintf(stderr, "Warning: Space token ' ' not found in vocab for dummy prefix.\n");
        }
    }

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) { // Not a continuation byte
            str_len = 0;
        }

        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';

        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) { // If next is continuation and buffer not full for a char
            continue;
        }

        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0;
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            // Ensure tokens[i] and tokens[i+1] are valid indices
            if (tokens[i] < 0 || tokens[i] >= t->vocab_size || 
                tokens[i+1] < 0 || tokens[i+1] >= t->vocab_size) {
                // This shouldn't happen if encoding is correct
                // fprintf(stderr, "Warning: Invalid token index during merge step.\n");
                continue; 
            }
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        tokens[best_idx] = best_id;
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare_prob_index(const void* a, const void* b) { // Renamed from `compare` to avoid conflict
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare_prob_index); // Use renamed compare function

    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; 
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; 
        }
    }

    float r = coin * cumulative_prob; // Sample within the truncated set
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; 
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
    if(!sampler->probindex) {fprintf(stderr, "malloc failed for probindex in sampler\n"); exit(EXIT_FAILURE);}
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { 
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    int next;
    if (sampler->temperature == 0.0f) {
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        softmax(logits, sampler->vocab_size);
        float coin = random_f32(&sampler->rng_state);
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
#if defined(_WIN32)
    // Windows specific timing. This is a placeholder, win.h might provide a better one.
    // If win.h provides QueryPerformanceCounter, that's preferred.
    // For simplicity, using clock() as a fallback if no high-resolution timer is in win.h
    return (long)((double)clock() * 1000.0 / CLOCKS_PER_SEC);
#elif defined(__linux__) || defined(__APPLE__) // Or other POSIX systems
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
#else
    // Fallback for other systems (might be less precise)
    // This is a common fallback if clock_gettime is not available.
    // Note: clock() measures CPU time, not wall-clock time on some systems.
    return (long)((double)clock() * 1000.0 / CLOCKS_PER_SEC);
#endif
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    int num_prompt_tokens = 0;
    size_t prompt_len = strlen(prompt);
    int* prompt_tokens = (int*)malloc((prompt_len + 3) * sizeof(int)); 
    if(!prompt_tokens) {fprintf(stderr, "malloc failed for prompt_tokens in generate\n"); exit(EXIT_FAILURE);}

    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        free(prompt_tokens); // Free memory before exit
        exit(EXIT_FAILURE);
    }

    long start = 0; 
    int next;      
    int token = prompt_tokens[0]; 
    int pos = 0;   
    while (pos < steps) {
        float* logits = forward(transformer, token, pos);

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(sampler, logits);
        }
        pos++;

        if (next == 1 || next == 2) { break; } // BOS or EOS token often used as sequence end

        char* piece = decode(tokenizer, token, next);
        safe_printf(piece); 
        fflush(stdout);
        token = next;

        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    if (pos > 1) {
        long end = time_in_ms();
        if (end > start) { // Avoid division by zero or negative time
             fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
        } else {
             fprintf(stderr, "achieved tok/s: (timing error or too fast)\n");
        }
    }
    free(prompt_tokens);
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; 
        }
    }
}

// ----------------------------------------------------------------------------
// chat loop

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps) {

    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152]; // Ensure this is large enough
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(1152 * sizeof(int)); // Corresponds to rendered_prompt size
    if(!prompt_tokens) {fprintf(stderr, "malloc failed for prompt_tokens in chat\n"); exit(EXIT_FAILURE);}
    int user_idx;

    int8_t user_turn = 1; 
    int next = 0; // Initialize next
    int token = 0; // Initialize token
    int pos = 0;   
    while (pos < steps) {
        if (user_turn) {
            if (pos == 0) {
                if (cli_system_prompt == NULL) {
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                } else {
                    strncpy(system_prompt, cli_system_prompt, sizeof(system_prompt) - 1);
                    system_prompt[sizeof(system_prompt) - 1] = '\0';
                }
            }
            if (pos == 0 && cli_user_prompt != NULL) {
                strncpy(user_prompt, cli_user_prompt, sizeof(user_prompt) - 1);
                user_prompt[sizeof(user_prompt)-1] = '\0';
            } else {
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }

            if (pos == 0 && system_prompt[0] != '\0') {
                // Ensure rendered_prompt is large enough for the formatted string.
                // snprintf is safer than sprintf.
                snprintf(rendered_prompt, sizeof(rendered_prompt), "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]", system_prompt, user_prompt);
            } else {
                snprintf(rendered_prompt, sizeof(rendered_prompt), "[INST] %s [/INST]", user_prompt);
            }
            
            encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            if (num_prompt_tokens < 1 && strlen(rendered_prompt) > 0) { // Check if encoding produced tokens
                fprintf(stderr, "Warning: encoding of rendered prompt produced 0 tokens. Prompt: '%s'\n", rendered_prompt);
                // Potentially skip this turn or handle error, for now, we might loop if user_idx isn't advanced.
            }
            user_idx = 0; 
            user_turn = 0;
            printf("Assistant: ");
            fflush(stdout); // Ensure "Assistant: " is printed before potential model output
            if (num_prompt_tokens == 0) continue; // If no tokens from user, wait for next input
            token = prompt_tokens[user_idx++]; // Prime the first token
        } else { // Assistant's turn or processing prompt
             if (user_idx < num_prompt_tokens) {
                token = prompt_tokens[user_idx++];
            } else { // Assistant generating
                token = next;
            }
        }
        
        // EOS (=2) token ends the Assistant turn (or BOS for next sequence)
        // llama models often use EOS (token 2) to signify end of assistant's turn.
        if (token == 2 && !user_turn) { // If assistant emits EOS, it's user's turn
            user_turn = 1;
            printf("\n"); // Newline after assistant's response
            // Don't increment pos here, as this EOS isn't "processed" by the model for next token.
            // The loop will restart, get new user input, and pos will increment after forward().
            continue; 
        }
        if (token == 1 && pos > 0 && !user_turn) { // BOS might also indicate end of sequence for assistant
            user_turn = 1;
            printf("\n");
            continue;
        }


        float* logits = forward(transformer, token, pos);
        next = sample(sampler, logits);
        pos++;

        // Only print if assistant is generating (i.e., past the initial prompt)
        // and the next token is not EOS (which is handled above to switch turn)
        if (user_idx >= num_prompt_tokens && next != 2 && next != 1 && !user_turn) {
            char* piece = decode(tokenizer, token, next);
            safe_printf(piece); 
            fflush(stdout);
        }
         // If the *sampled* next token is EOS, it will be handled at the start of the next iteration
         // to switch to user_turn.
    }
    printf("\n");
    free(prompt_tokens);
}


// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

    // default parameters
    char *checkpoint_path = NULL;  // e.g. out/model.bin
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    char *prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    char *mode = "generate";    // generate|chat
    char *system_prompt = NULL; // the (optional) system prompt to use in chat mode

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        if (i + 1 >= argc) { error_usage(); } 
        if (argv[i][0] != '-') { error_usage(); } 
        if (strlen(argv[i]) != 2) { error_usage(); } 
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = strtoull(argv[i + 1], NULL, 10); } // Use strtoull for unsigned long long
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else { error_usage(); }
    }

    // parameter validation/overrides
    if (rng_seed == 0) rng_seed = (unsigned long long)time(NULL); // Cast time(NULL)
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9; // Ensure topp is within [0,1] or use default
    if (steps <= 0) steps = 0; // steps = 0 means use seq_len

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // override to ~max length

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // run!
    if (strcmp(mode, "generate") == 0) {
        generate(&transformer, &tokenizer, &sampler, prompt, steps);
    } else if (strcmp(mode, "chat") == 0) {
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
#endif
