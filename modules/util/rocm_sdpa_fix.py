"""Routes attention with head sizes above 256 to PyTorch's math kernel on RDNA3 (gfx11) GPUs under ROCm.

PyTorch 2.13's flash and memory-efficient attention on ROCm (AOTriton 0.12b) give wrong outputs and NaN gradients
there for a head size of 512, in fp32, bf16 and fp16 (measured on an RX 7900 XTX with scripts/rocm_check.py: relative
errors of 1 to 17, while the math kernel matches a float64 reference). PyTorch limits gfx11 to a head size of 256 only
for AOTriton 0.11 (aotriton_max_hdim in aten/src/ATen/native/transformers/cuda/sdp_utils.cpp), so 0.12b gets no limit.
Every VAE (SD1.5 to Flux/Chroma) has one 512-wide attention head in its middle block, so without this every latent
cached and every image decoded on such a GPU was wrong (the Flux/Chroma ae.safetensors latent by 9%, a randomly
initialized VAE's by 25-40%).

install() replaces torch.nn.functional.scaled_dot_product_attention, which diffusers and transformers look up on
every call. Other calls go straight to PyTorch's function.
"""
import functools

import torch
import torch.nn.functional as F

original_scaled_dot_product_attention = F.scaled_dot_product_attention

MAX_HEAD_DIM = 256
# the math kernel builds the query x key score matrix: split long queries so it (with its softmax) stays near this
_CHUNK_BYTES = 512 * 2**20

_affected_devices: dict[int, bool] = {}


def applies_to(device: torch.device) -> bool:
    """whether head sizes above 256 are routed to the math kernel on this device"""
    if torch.version.hip is None or device.type != "cuda" or not torch.cuda.is_available():
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _affected_devices:
        _affected_devices[index] = torch.cuda.get_device_properties(index).gcnArchName.startswith("gfx11")
    return _affected_devices[index]


def _needs_math(query: torch.Tensor) -> bool:
    return query.shape[-1] > MAX_HEAD_DIM and applies_to(query.device)


def _math(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa):
    # PyTorch's math kernel called directly. torch.nn.attention.sdpa_kernel would select it through process-global
    # backend flags, and two threads overlapping in it (caching runs the VAE in worker threads) left flash and
    # memory-efficient attention switched off for the rest of the process.
    return torch.ops.aten._scaled_dot_product_attention_math(
        query, key, value, attn_mask, dropout_p, is_causal, scale=scale, enable_gqa=enable_gqa)[0]


def math_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    """PyTorch's math attention kernel, split along the query length when the score matrix would be large"""
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        # what scaled_dot_product_attention does before it runs the math kernel
        attn_mask = torch.zeros(attn_mask.shape, dtype=query.dtype, device=attn_mask.device) \
            .masked_fill(attn_mask.logical_not(), float("-inf"))
    rows = query.shape[-2]
    batch_heads = query[..., 0, 0].numel()
    chunk = max(1, _CHUNK_BYTES // max(1, batch_heads * key.shape[-2] * 4 * 3))  # fp32 scores, probs, grads
    if chunk >= rows or is_causal or dropout_p > 0.0:
        return _math(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)
    outputs = []
    for start in range(0, rows, chunk):
        mask = attn_mask
        if mask is not None and mask.dim() >= 2 and mask.shape[-2] != 1:
            mask = mask[..., start:start + chunk, :]
        outputs.append(_math(query[..., start:start + chunk, :], key, value, mask, 0.0, False, scale, enable_gqa))
    return torch.cat(outputs, dim=-2)


@functools.wraps(original_scaled_dot_product_attention)
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
                                 enable_gqa=False):
    if _needs_math(query):
        return math_attention(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)
    return original_scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                                                 is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)


def install():
    F.scaled_dot_product_attention = scaled_dot_product_attention


def uninstall():
    F.scaled_dot_product_attention = original_scaled_dot_product_attention


def installed() -> bool:
    return F.scaled_dot_product_attention is scaled_dot_product_attention
