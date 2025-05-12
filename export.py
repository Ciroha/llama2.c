"""
This script has functions and utilties for model export.
Basically, we have a bunch of versions of the model, and we
want to export them to .bin files to be read from and inferenced in C.

Among the "input" versions of PyTorch files/models:
- Official Llama 2 weights released by Meta
- Huggingface weights available on the hub
- llama2.c (this repo) trained models

Among the "output" versions of .bin files:
- v0: Legacy files of the original llama2.c repo (will eventually be DEPRECATED)
- v1-vN: Improved .bin files with a proper header, cache alignment, etc.

This script aspires to provide all of these conversions.
"""
import os
import gzip
import shutil
import struct
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from model import ModelArgs, Transformer

# -----------------------------------------------------------------------------
# common utilities

def serialize_fp32(file, tensor):
    """ writes one fp32 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)

def serialize_int8(file, tensor):
    """ writes one int8 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).numpy().astype(np.int8)
    b = struct.pack(f'{len(d)}b', *d)
    file.write(b)

def quantize_q80(w, group_size):
    """
    takes a tensor and returns the Q8_0 quantized version
    i.e. symmetric quantization into int8, range [-127,127]
    """
    assert w.numel() % group_size == 0
    ori_shape = w.shape
    w = w.float() # convert to float32
    w = w.reshape(-1, group_size)
    # find the max in each group
    wmax = torch.abs(w).max(dim=1).values
    # calculate the scaling factor such that float = quant * scale
    scale = wmax / 127.0
    # scale into range [-127, 127]
    quant = w / scale[:,None]
    # round to nearest integer
    int8val = torch.round(quant).to(torch.int8)
    # dequantize by rescaling
    fp32val = (int8val.float() * scale[:,None]).view(-1)
    fp32valr = fp32val.reshape(-1, group_size)
    # calculate the max error in each group
    err = torch.abs(fp32valr - w).max(dim=1).values
    # find the max error across all groups
    maxerr = err.max().item()
    return int8val, scale, maxerr

# -----------------------------------------------------------------------------
# legacy

def legacy_export(model, filepath):
    """ Original export of llama2.c bin files, i.e. version v0 """
    out_file = open(filepath, 'wb')

    # first write out the header
    hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    p = model.params
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
    # legacy format uses negative/positive vocab size as a shared classifier flag
    if not shared_classifier:
        p.vocab_size = -p.vocab_size
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                                    n_kv_heads, p.vocab_size, p.max_seq_len)
    out_file.write(header)

    # next write out the embedding weights
    serialize_fp32(out_file, model.tok_embeddings.weight)

    # now all the layers
    # attention weights
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention_norm.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention.wq.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention.wk.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention.wv.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention.wo.weight)
    # ffn weights
    for layer in model.layers:
        serialize_fp32(out_file, layer.ffn_norm.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.feed_forward.w1.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.feed_forward.w2.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.feed_forward.w3.weight)
    # final rmsnorm
    serialize_fp32(out_file, model.norm.weight)
    # freqs_cis
    serialize_fp32(out_file, model.freqs_cos[:p.max_seq_len])
    serialize_fp32(out_file, model.freqs_sin[:p.max_seq_len])

    # final classifier weights
    if not shared_classifier:
        serialize_fp32(out_file, model.output.weight)

    # write to binary file
    out_file.close()
    print(f"wrote {filepath}")

# -----------------------------------------------------------------------------
# new version

def version1_export(model, filepath):
    """
    Export the model weights in full float32 .bin file to be read from C.
    This is same as legacy_export, but with a proper header.
    """
    version = 1

    out_file = open(filepath, 'wb')
    # first write out the header. the header will be 256 bytes
    # 1) write magic, which will be uint32 of "ak42" in ASCII
    out_file.write(struct.pack('I', 0x616b3432))
    # 2) write version, which will be int
    out_file.write(struct.pack('i', version))
    # 3) write the params, which will be 7 ints
    p = model.params
    hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                                    n_kv_heads, p.vocab_size, p.max_seq_len)
    out_file.write(header)
    # 4) write some other flags
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
    out_file.write(struct.pack('B', int(shared_classifier)))
    pad = 256 - out_file.tell() # pad rest with zeros; tell returns current pos
    assert pad >= 0
    out_file.write(b'\0' * pad)

    # now let's write out all the params
    weights = [
        *[layer.attention_norm.weight for layer in model.layers],
        *[layer.ffn_norm.weight for layer in model.layers],
        model.norm.weight,
        model.tok_embeddings.weight,
        *[layer.attention.wq.weight for layer in model.layers],
        *[layer.attention.wk.weight for layer in model.layers],
        *[layer.attention.wv.weight for layer in model.layers],
        *[layer.attention.wo.weight for layer in model.layers],
        *[layer.feed_forward.w1.weight for layer in model.layers],
        *[layer.feed_forward.w2.weight for layer in model.layers],
        *[layer.feed_forward.w3.weight for layer in model.layers],
    ]
    if not shared_classifier:
        weights.append(model.output.weight)
    for w in weights:
        serialize_fp32(out_file, w)

    # write to binary file
    out_file.close()
    print(f"wrote {filepath}")

def version2_export(model, filepath, group_size=64):
    """
    Export the model weights in Q8_0 into .bin file to be read from C.
    That is:
    - quantize all weights to symmetric int8, in range [-127, 127]
    - all other tensors (the rmsnorm params) are kept and exported in fp32
    - quantization is done in groups of group_size to reduce the effects of any outliers
    """
    version = 2
    print(f"\n--- Starting Export to Version {version} Quantized Model File: {filepath} ---")
    print(f"Using group_size: {group_size}")

    # let's first do some validation for this export type
    while model.params.dim % group_size != 0:
        original_group_size = group_size
        group_size //= 2
        print(f"WARNING: model.params.dim ({model.params.dim}) is not divisible by group_size ({original_group_size}).")
        print(f"BACKOFF: reducing group_size to {group_size} to fit hidden_dim.")
        if group_size == 0:
            raise ValueError("group_size became 0, cannot proceed.")

    p = model.params
    weight_names = []
    # Token Embeddings
    weight_names.append("tok_embeddings.weight")
    # Attention weights per layer
    for i in range(p.n_layers):
        weight_names.append(f"layers.{i}.attention.wq.weight")
    for i in range(p.n_layers):
        weight_names.append(f"layers.{i}.attention.wk.weight")
    for i in range(p.n_layers):
        weight_names.append(f"layers.{i}.attention.wv.weight")
    for i in range(p.n_layers):
        weight_names.append(f"layers.{i}.attention.wo.weight")
    # FFN weights per layer
    for i in range(p.n_layers):
        weight_names.append(f"layers.{i}.feed_forward.w1.weight")
    for i in range(p.n_layers):
        weight_names.append(f"layers.{i}.feed_forward.w2.weight")
    for i in range(p.n_layers):
        weight_names.append(f"layers.{i}.feed_forward.w3.weight")

    weights_to_quantize = [
        model.tok_embeddings.weight,
        *[layer.attention.wq.weight for layer in model.layers],
        *[layer.attention.wk.weight for layer in model.layers],
        *[layer.attention.wv.weight for layer in model.layers],
        *[layer.attention.wo.weight for layer in model.layers],
        *[layer.feed_forward.w1.weight for layer in model.layers],
        *[layer.feed_forward.w2.weight for layer in model.layers],
        *[layer.feed_forward.w3.weight for layer in model.layers],
    ]
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
    if not shared_classifier:
        weights_to_quantize.append(model.output.weight)
        weight_names.append("output.weight")

    for w_idx, w in enumerate(weights_to_quantize):
        if w.numel() % group_size != 0:
            # This should ideally not happen if dim is divisible by group_size and weights are standard
            print(f"WARNING: Weight {weight_names[w_idx]} (shape: {w.shape}, numel: {w.numel()}) "+
                  f"is not perfectly divisible by group_size {group_size}.")
            # Fallback or error based on strictness, for now, we'll let it proceed if quantize_q80 handles it by padding/erroring
            # assert w.numel() % group_size == 0, f"weight {weight_names[w_idx]} has numel {w.numel()}, not a multiple of group_size {group_size}"


    out_file = open(filepath, 'wb')
    current_offset = 0
    print(f"\n--- Writing File Header (256 bytes) ---")
    # 1) write magic, which will be uint32 of "ak42" in ASCII
    start_offset = out_file.tell()
    out_file.write(struct.pack('I', 0x616b3432))
    print(f"  Magic Number (0x616b3432): Offset {start_offset}-{out_file.tell()-1}, Size: {out_file.tell() - start_offset} bytes")
    # 2) write version, which will be int
    start_offset = out_file.tell()
    out_file.write(struct.pack('i', version))
    print(f"  Version ({version}): Offset {start_offset}-{out_file.tell()-1}, Size: {out_file.tell() - start_offset} bytes")
    # 3) write the params, which will be 7 ints
    start_offset = out_file.tell()
    hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    header_params = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                                    n_kv_heads, p.vocab_size, p.max_seq_len)
    out_file.write(header_params)
    print(f"  Model Params (dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len): Offset {start_offset}-{out_file.tell()-1}, Size: {out_file.tell() - start_offset} bytes")
    # 4) write some other flags
    start_offset = out_file.tell()
    out_file.write(struct.pack('B', int(shared_classifier)))
    print(f"  Shared Classifier Flag ({int(shared_classifier)}): Offset {start_offset}-{out_file.tell()-1}, Size: {out_file.tell() - start_offset} byte")
    start_offset = out_file.tell()
    out_file.write(struct.pack('i', group_size)) # group size used for quantization
    print(f"  Group Size ({group_size}): Offset {start_offset}-{out_file.tell()-1}, Size: {out_file.tell() - start_offset} bytes")
    
    header_end_offset = out_file.tell()
    pad_bytes = 256 - header_end_offset 
    assert pad_bytes >= 0
    if pad_bytes > 0:
        out_file.write(b'\0' * pad_bytes)
        print(f"  Padding: Offset {header_end_offset}-{out_file.tell()-1}, Size: {pad_bytes} bytes")
    
    total_header_size = out_file.tell()
    print(f"Total Header Size: {total_header_size} bytes (Expected: 256 bytes)")
    assert total_header_size == 256, "Header size mismatch!"
    current_offset = total_header_size

    print(f"\n--- Writing FP32 RMSNorm Weights ---")
    # first let's write out all the params that we are keeping in fp32: the norms
    rms_weights_info = []
    for i, layer in enumerate(model.layers): # attention norms
        name = f"layers.{i}.attention_norm.weight"
        start_offset = out_file.tell()
        serialize_fp32(out_file, layer.attention_norm.weight)
        end_offset = out_file.tell()
        size = end_offset - start_offset
        rms_weights_info.append({'name': name, 'start': start_offset, 'end': end_offset -1, 'size': size, 'shape': tuple(layer.attention_norm.weight.shape), 'dtype': 'fp32'})
        current_offset = end_offset
    for i, layer in enumerate(model.layers): # MLP norms
        name = f"layers.{i}.ffn_norm.weight"
        start_offset = out_file.tell()
        serialize_fp32(out_file, layer.ffn_norm.weight)
        end_offset = out_file.tell()
        size = end_offset - start_offset
        rms_weights_info.append({'name': name, 'start': start_offset, 'end': end_offset -1, 'size': size, 'shape': tuple(layer.ffn_norm.weight.shape), 'dtype': 'fp32'})
        current_offset = end_offset
    
    name = "model.norm.weight" # final pre-classifier norm
    start_offset = out_file.tell()
    serialize_fp32(out_file, model.norm.weight)
    end_offset = out_file.tell()
    size = end_offset - start_offset
    rms_weights_info.append({'name': name, 'start': start_offset, 'end': end_offset -1, 'size': size, 'shape': tuple(model.norm.weight.shape), 'dtype': 'fp32'})
    current_offset = end_offset

    for info in rms_weights_info:
        print(f"  Written: {info['name']} (Shape: {info['shape']}, Dtype: {info['dtype']})")
        print(f"    File Offset: {info['start']}-{info['end']}, Size: {info['size']} bytes")


    print(f"\n--- Quantizing and Writing Q8_0 Weights (Group Size: {group_size}) ---")
    ew = [] # To store (error, shape, name)
    quantized_weights_layout_info = []

    for i, w in enumerate(weights_to_quantize):
        weight_name = weight_names[i]
        original_dtype = w.dtype
        
        # quantize this weight
        q, s, err = quantize_q80(w, group_size)
        
        # Log before writing
        print(f"  ({i+1}/{len(weights_to_quantize)}) Processing: {weight_name}")
        print(f"    Original Shape: {tuple(w.shape)}, Original Dtype: {original_dtype}")
        print(f"    Quantized (INT8) Shape: {tuple(q.shape)}")
        print(f"    Scales (FP32) Count: {s.numel()}, Shape: {tuple(s.shape)}")
        print(f"    Max Quantization Error for this tensor: {err:.6f}")

        # save the int8 weights to file
        start_offset_q = out_file.tell()
        serialize_int8(out_file, q) 
        end_offset_q = out_file.tell()
        size_q = end_offset_q - start_offset_q
        quantized_weights_layout_info.append({
            'name': f"{weight_name} (INT8 values)", 
            'start': start_offset_q, 'end': end_offset_q -1, 'size': size_q, 
            'shape': tuple(q.shape), 'dtype': 'int8'
        })
        print(f"    Written INT8 values: File Offset {start_offset_q}-{end_offset_q-1}, Size: {size_q} bytes")
        current_offset = end_offset_q

        # save scale factors
        start_offset_s = out_file.tell()
        serialize_fp32(out_file, s) 
        end_offset_s = out_file.tell()
        size_s = end_offset_s - start_offset_s
        quantized_weights_layout_info.append({
            'name': f"{weight_name} (FP32 scales)", 
            'start': start_offset_s, 'end': end_offset_s -1, 'size': size_s, 
            'shape': tuple(s.shape), 'dtype': 'fp32'
        })
        print(f"    Written FP32 scales: File Offset {start_offset_s}-{end_offset_s-1}, Size: {size_s} bytes")
        current_offset = end_offset_s
        
        ew.append((err, w.shape, weight_name))

    # print the highest error across all weights
    ew.sort(key=lambda x: x[0], reverse=True) # Sort by error
    if ew:
        print(f"\nMax quantization group error across all weights: {ew[0][0]:.6f} (for tensor: {ew[0][2]}, shape: {ew[0][1]})")
        print("Min quantization group error across all weights: {:.6f} (for tensor: {}, shape: {})".format(ew[-1][0], ew[-1][2], ew[-1][1]))

    out_file.close()
    final_file_size = os.path.getsize(filepath)
    print(f"\n--- Export Complete ---")
    print(f"Wrote quantized model to: {filepath}")
    print(f"Final file size: {final_file_size} bytes.")
    if final_file_size != current_offset:
        print(f"WARNING: Final file size ({final_file_size}) does not match tracked offset ({current_offset}). Check logic.")

def hf_export(llama_model, filepath, group_size=64, dtype=torch.float32):
    """ Generate the pytorch_model.bin state_dict and config.json for HuggingFace """

    try:
        from transformers.models.llama.configuration_llama import LlamaConfig
    except ImportError:
        print("Error: transformers package is required to load huggingface models")
        print("Please run `pip install transformers` to install it")
        return None

    # Generate LlamaModel state_dict
    hf_state_dict = {}

    # Sometimes we have repeated key values for the heads
    dim = llama_model.params.dim
    num_key_value_heads = llama_model.params.n_kv_heads
    n_rep = llama_model.params.n_heads // num_key_value_heads
    key_value_dim = dim // n_rep

    # HuggingFace needs the weights permuted.
    # See: https://github.com/huggingface/transformers/blob/b132c1703eb1c8bd9dfa4ad6a9be2bfd6ef819e9/src/transformers/models/llama/convert_llama_weights_to_hf.py#L122
    def permute_original(w, n_heads=llama_model.params.n_heads, dim1=dim, dim2=dim):
        return w.view(dim1, dim2).reshape(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    # Transfer weights from llama model to the HF state dictionary format
    hf_state_dict['model.embed_tokens.weight'] = llama_model.tok_embeddings.weight.clone().to(dtype)
    hf_state_dict['model.norm.weight'] = llama_model.norm.weight.clone().to(dtype)

    # Add each layer's weights to the HF state dictionary
    for i, layer in enumerate(llama_model.layers):
        layer_id = layer.layer_id
        hf_state_dict[f'model.layers.{i}.input_layernorm.weight'] = llama_model.layers[layer_id].attention_norm.weight.clone().to(dtype)
        hf_state_dict[f'model.layers.{i}.self_attn.q_proj.weight'] = permute_original(llama_model.layers[layer_id].attention.wq.weight.clone()).to(dtype)
        hf_state_dict[f'model.layers.{i}.self_attn.k_proj.weight'] = permute_original(llama_model.layers[layer_id].attention.wk.weight.clone(), num_key_value_heads, key_value_dim, dim).to(dtype)
        hf_state_dict[f'model.layers.{i}.self_attn.v_proj.weight'] = llama_model.layers[layer_id].attention.wv.weight.clone().to(dtype)
        hf_state_dict[f'model.layers.{i}.self_attn.o_proj.weight'] = llama_model.layers[layer_id].attention.wo.weight.clone().to(dtype)
        hf_state_dict[f'model.layers.{i}.post_attention_layernorm.weight'] = llama_model.layers[layer_id].ffn_norm.weight.clone().to(dtype)
        hf_state_dict[f'model.layers.{i}.mlp.gate_proj.weight'] = llama_model.layers[layer_id].feed_forward.w1.weight.clone().to(dtype)
        hf_state_dict[f'model.layers.{i}.mlp.down_proj.weight'] = llama_model.layers[layer_id].feed_forward.w2.weight.clone().to(dtype)
        hf_state_dict[f'model.layers.{i}.mlp.up_proj.weight'] = llama_model.layers[layer_id].feed_forward.w3.weight.clone().to(dtype)

    # llama2.c usually uses tied weights -> reference the embed_tokens.weights instead
    hf_state_dict['lm_head.weight'] = hf_state_dict['model.embed_tokens.weight']

    # We check that the embeddings are tied, else use manual output weights
    _embeddings_are_tied: bool = torch.equal(llama_model.tok_embeddings.weight, llama_model.output.weight)
    if not _embeddings_are_tied:
        hf_state_dict['lm_head.weight'] = llama_model.output.weight.clone().to(dtype)


    # Generate LlamaConfig (seen in transformers.models.llama.configuration_llama)

    # Extract necessary attributes from llama.c model
    vocab_size = llama_model.params.vocab_size
    hidden_size = llama_model.params.dim
    intermediate_size = llama_model.layers[0].feed_forward.w1.weight.shape[0]
    num_hidden_layers = llama_model.params.n_layers
    num_attention_heads = llama_model.params.n_heads
    num_key_value_heads = llama_model.params.n_kv_heads
    max_position_embeddings = llama_model.params.max_seq_len
    rms_norm_eps = llama_model.params.norm_eps

    # TODO check values for:
    # pretraining_tp, initializer_range, use_cache,
    # rope_theta, and rope_scaling.

    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=rms_norm_eps,
        tie_word_embeddings=_embeddings_are_tied,
        # Manual
        architectures=["LlamaForCausalLM"],
        hidden_act="silu",
    )


    # Save files in directory filepath
    # First make the directory if it doesn't exist
    os.makedirs(filepath, exist_ok=True)

    # Save the state dictionary in .bin format, and config as .json
    torch.save(hf_state_dict, os.path.join(filepath, "pytorch_model.bin"))
    config.save_pretrained(filepath)


# -----------------------------------------------------------------------------
# Load / import functions

def load_checkpoint(checkpoint):

    # load the provided model checkpoint
    checkpoint_dict = torch.load(checkpoint, map_location='cpu')
    gptconf = ModelArgs(**checkpoint_dict['model_args'])
    model = Transformer(gptconf)
    state_dict = checkpoint_dict['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def load_meta_model(model_path):
    params_path = os.path.join(model_path, 'params.json')
    with open(params_path) as f:
        params = json.load(f)
        print(params)

    model_paths = sorted(list(Path(model_path).glob('consolidated.*.pth')))
    models = [torch.load(p, map_location='cpu') for p in model_paths]

    def concat_weights(models):
        state_dict = {}
        for name in list(models[0]):
            tensors = [model[name] for model in models]
            if len(tensors) == 1 or len(tensors[0].shape) == 1:
                state_dict[name] = tensors[0]
                continue
            is_axis_1 = (
                name.startswith('tok_embeddings.')
                or name.endswith('.attention.wo.weight')
                or name.endswith('.feed_forward.w2.weight')
            )
            axis = 1 if is_axis_1 else 0
            state_dict[name] = torch.cat(tensors, dim=axis)
            for model in models:
                del model[name]
        return state_dict

    state_dict = concat_weights(models)
    del models

    # set ModelArgs
    config = ModelArgs()
    config.dim = params["dim"]
    config.n_layers = params["n_layers"]
    config.n_heads = params["n_heads"]
    config.n_kv_heads = params.get('n_kv_heads') or params['n_heads']
    config.multiple_of = params["multiple_of"]
    config.norm_eps = params["norm_eps"]

    config.vocab_size = state_dict['tok_embeddings.weight'].shape[0]
    config.max_seq_len = 2048


    # create a new Transformer object and set weights
    model = Transformer(config)

    model.tok_embeddings.weight = nn.Parameter(state_dict['tok_embeddings.weight'])
    model.norm.weight = nn.Parameter(state_dict['norm.weight'])

    for layer in model.layers:
        i = layer.layer_id
        layer.attention_norm.weight = nn.Parameter(state_dict[f'layers.{i}.attention_norm.weight'])
        layer.attention.wq.weight = nn.Parameter(state_dict[f'layers.{i}.attention.wq.weight'])
        layer.attention.wk.weight = nn.Parameter(state_dict[f'layers.{i}.attention.wk.weight'])
        layer.attention.wv.weight = nn.Parameter(state_dict[f'layers.{i}.attention.wv.weight'])
        layer.attention.wo.weight = nn.Parameter(state_dict[f'layers.{i}.attention.wo.weight'])
        layer.ffn_norm.weight = nn.Parameter(state_dict[f'layers.{i}.ffn_norm.weight'])
        layer.feed_forward.w1.weight = nn.Parameter(state_dict[f'layers.{i}.feed_forward.w1.weight'])
        layer.feed_forward.w2.weight = nn.Parameter(state_dict[f'layers.{i}.feed_forward.w2.weight'])
        layer.feed_forward.w3.weight = nn.Parameter(state_dict[f'layers.{i}.feed_forward.w3.weight'])

    # final classifier
    model.output.weight = nn.Parameter(state_dict['output.weight'])
    model.eval()
    return model

def load_hf_model(model_path):

    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        print("Error: transformers package is required to load huggingface models")
        print("Please run `pip install transformers` to install it")
        return None

    # load HF model
    hf_model = AutoModelForCausalLM.from_pretrained(model_path)
    hf_dict = hf_model.state_dict()

    # convert LlamaConfig to ModelArgs
    config = ModelArgs()
    config.dim = hf_model.config.hidden_size
    config.n_layers = hf_model.config.num_hidden_layers
    config.n_heads = hf_model.config.num_attention_heads
    config.n_kv_heads = hf_model.config.num_attention_heads
    config.vocab_size = hf_model.config.vocab_size
    config.hidden_dim = hf_model.config.intermediate_size
    config.norm_eps = hf_model.config.rms_norm_eps
    config.max_seq_len = hf_model.config.max_position_embeddings

    # create a new Transformer object and set weights
    model = Transformer(config)

    model.tok_embeddings.weight = nn.Parameter(hf_dict['model.embed_tokens.weight'])
    model.norm.weight = nn.Parameter(hf_dict['model.norm.weight'])

    # huggingface permutes WQ and WK, this function reverses it
    def permute_reverse(w, n_heads=config.n_heads, dim1=config.dim, dim2=config.dim):
        return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    for layer in model.layers:
        i = layer.layer_id
        layer.attention_norm.weight = nn.Parameter(hf_dict[f'model.layers.{i}.input_layernorm.weight'])
        layer.attention.wq.weight = nn.Parameter(permute_reverse(hf_dict[f'model.layers.{i}.self_attn.q_proj.weight']))
        layer.attention.wk.weight = nn.Parameter(permute_reverse(hf_dict[f'model.layers.{i}.self_attn.k_proj.weight']))
        layer.attention.wv.weight = nn.Parameter(hf_dict[f'model.layers.{i}.self_attn.v_proj.weight'])
        layer.attention.wo.weight = nn.Parameter(hf_dict[f'model.layers.{i}.self_attn.o_proj.weight'])
        layer.ffn_norm.weight = nn.Parameter(hf_dict[f'model.layers.{i}.post_attention_layernorm.weight'])
        layer.feed_forward.w1.weight = nn.Parameter(hf_dict[f'model.layers.{i}.mlp.gate_proj.weight'])
        layer.feed_forward.w2.weight = nn.Parameter(hf_dict[f'model.layers.{i}.mlp.down_proj.weight'])
        layer.feed_forward.w3.weight = nn.Parameter(hf_dict[f'model.layers.{i}.mlp.up_proj.weight'])

    # final classifier
    model.output.weight = nn.Parameter(hf_dict['lm_head.weight'])
    model.eval()
    return model


# -----------------------------------------------------------------------------
# API entrypoint

def model_export(model, filepath, version, dtype=torch.float32):
    """
    Versions docs:
    v-1:huggingface export, i.e. intended for use outside of this repo, in HF
    v0: legacy llama2.c float format, DEPRECATED
    v1: float32 export
    v2: int8 quantized Q8_0 export, similar to llama.cpp, in groups
    # TODO: add dtype export support for other versions (?)
    """
    if version == 0:
        legacy_export(model, filepath)
    elif version == 1:
        version1_export(model, filepath)
    elif version == 2:
        version2_export(model, filepath)
    elif version == -1:
        hf_export(model, filepath, dtype)
    else:
        raise ValueError(f"unknown version {version}")

def torchscript_export(model, filepath, zero_params=False, gzip_output=False):
    """
    (This was submitted via a PR earlier. Leaving it here, but "orphaned" for now)
    Saves the model as a TorchScript.
    The resulting file can be loaded in C++ code and then used for training or
    inference with:
        #include <torch/script.h>
        torch::jit::Module module = torch::jit::load("model.pt")
    Note that the serialized model includes the initial parameters and with the default
    ModelArgs the file is 59M and gzips down to 55M. If you want to serialize/distribute
    the model parameters separately you can zero out the parameters before saving it and
    it will gzip down to 780K.
    """

    # If requested zero params before saving the model. This is useful in
    # conjunction with gzip_output.
    if zero_params:
        for p in model.parameters():
            p.detach().zero_()

    torch.jit.save(torch.jit.script(model), filepath)

    if gzip_output:
        with open(filepath, "rb") as f_in:
            with gzip.open(f"{filepath}.gz", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.unlink(filepath)

# -----------------------------------------------------------------------------
# CLI entrypoint

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="the output filepath")
    parser.add_argument("--version", default=0, type=int, help="the version to export with")
    parser.add_argument("--dtype", type=str, help="dtype of the model (fp16, fp32)", default="fp32")
    parser.add_argument("--group_size", type=int, default=64, help="group size for quantization (version 2)") # 新增参数
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, help="model checkpoint, .pt file")
    group.add_argument("--meta-llama", type=str, help="meta llama model path")
    group.add_argument("--hf", type=str, help="huggingface model path")
    args = parser.parse_args()
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    if args.checkpoint:
        model = load_checkpoint(args.checkpoint)
    elif args.meta_llama:
        model = load_meta_model(args.meta_llama)
    elif args.hf:
        model = load_hf_model(args.hf)

    if model is None:
        parser.error("Can't load input model!")

    # export
    if args.version == 2:
        print(f"Exporting version 2 with group_size: {args.group_size}")
        version2_export(model, args.filepath, group_size=args.group_size)
    else:
        model_export(model, args.filepath, args.version, args.dtype)
