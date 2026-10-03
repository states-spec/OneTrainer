import contextlib
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path

from modules.util.rocm_check.framework import Ctx, Rec, check
from modules.util.torch_util import fp8_matmul_supported

import torch

SECTION = "1 environment"

ENV_VARS = [
    "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HSA_OVERRIDE_GFX_VERSION",
    "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_ALLOC_CONF", "PYTORCH_TUNABLEOP_ENABLED",
    "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "TORCH_BLAS_PREFER_HIPBLASLT", "MIOPEN_FIND_MODE",
    "MIOPEN_USER_DB_PATH", "OT_CUDA_LOWMEM_MODE", "OT_EXPANDABLE_SEGMENTS",
]

PACKAGES = ["diffusers", "transformers", "accelerate", "safetensors", "bitsandbytes", "adv_optm",
            "prodigy-plus-schedule-free", "triton", "pytorch-triton-rocm", "mgds", "onnxruntime"]

# AOTriton (PyTorch's flash / memory-efficient attention on ROCm) treats these as experimental in the version torch 2.13
# ships (0.12b): without TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1, SDPA runs the slow math kernel on them
AOTRITON_EXPERIMENTAL = {"gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1152", "gfx1153", "gfx1200", "gfx1250"}

# torch._scaled_mm (fp8 matmul) is allowed on these ROCm architectures
FP8_ARCHS_ROCM = {"gfx942", "gfx950", "gfx1200", "gfx1201"}


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10,
                              cwd=Path(__file__).resolve().parents[3]).stdout.strip()
    except Exception:
        return "?"


@check(SECTION, "PyTorch, GPU, libraries and settings")
def environment(ctx: Ctx, rec: Rec):
    rec.info()
    rec.line(f"OneTrainer {_git('rev-parse', '--abbrev-ref', 'HEAD')} @ {_git('rev-parse', '--short', 'HEAD')}"
             f"{' (uncommitted changes)' if _git('status', '--porcelain', '--untracked-files=no') else ''}")
    rec.line(f"Python {sys.version.split()[0]}, torch {torch.__version__}, "
             f"HIP {torch.version.hip}, CUDA {torch.version.cuda}")
    rec.metric("torch", torch.__version__)
    rec.metric("hip", torch.version.hip)

    if ctx.is_gpu:
        props = torch.cuda.get_device_properties(ctx.device)
        free, total = torch.cuda.mem_get_info(ctx.device)
        rec.line(f"GPU: {props.name}, {ctx.arch}, {props.multi_processor_count} CUs/SMs, "
                 f"{total / 2**30:.1f} GiB ({free / 2**30:.1f} GiB free)")
        rec.metric("gpu", props.name)
        rec.metric("arch", ctx.arch)
        rec.metric("vram_gib", total / 2**30)
        rec.metric("vram_free_gib", free / 2**30)
        rec.line(f"GPUs visible: {torch.cuda.device_count()}, bf16 supported: {torch.cuda.is_bf16_supported()}")
        rec.line(f"BLAS library: {torch.backends.cuda.preferred_blas_library()}")
        if ctx.is_rocm:
            rec.line(f"flash attention library: {torch.backends.cuda.preferred_rocm_fa_library()}")
            if ctx.arch in AOTRITON_EXPERIMENTAL and os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL") != "1":
                rec.warn(f"{ctx.arch} is experimental in AOTriton: set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1, or "
                         f"SDPA uses the slow math kernel (see the SDPA checks)")
        try:
            import triton
            target = triton.runtime.driver.active.get_current_target()
            rec.line(f"Triton {triton.__version__}, target {target.backend} {target.arch}")
        except Exception as e:
            rec.line(f"Triton: not usable ({type(e).__name__}: {e})")
    else:
        rec.line(f"no GPU: running on {ctx.device} (torch.cuda.is_available() = {torch.cuda.is_available()})")

    versions = []
    for name in PACKAGES:
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            versions.append(f"{name} {importlib.metadata.version(name)}")
    rec.line("packages: " + ", ".join(versions))

    try:
        import bitsandbytes.cextension as bnb_ext
        if isinstance(bnb_ext.lib, bnb_ext.ErrorHandlerMockBNBNativeLibrary):
            rec.warn("bitsandbytes' native library failed to load: *_8BIT optimizers, INT_8 and NFLOAT_4 won't work")
        else:
            rec.line(f"bitsandbytes library: {os.path.basename(bnb_ext.lib._lib._name)}")
    except Exception as e:
        rec.warn(f"bitsandbytes import failed: {type(e).__name__}: {e}")

    rocm_version = Path("/opt/rocm/.info/version")
    if rocm_version.is_file():
        rec.line(f"system ROCm (/opt/rocm): {rocm_version.read_text().strip()}")
    env = [f"{name}={os.environ[name]}" for name in ENV_VARS if name in os.environ]
    rec.line("environment: " + (", ".join(env) if env else "none of the GPU variables are set"))
    alloc_conf = next((os.environ[n] for n in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_ALLOC_CONF")
                        if os.environ.get(n)), "")
    if ctx.is_rocm and "expandable_segments:true" in alloc_conf.replace(" ", "").lower():
        rec.warn("expandable_segments is on, so every check in this run uses it; compare with a run without it "
                 "(the allocator check does both)")


@check(SECTION, "fp8 matmul support (float W8A8 weight types)")
def fp8_support(ctx: Ctx, rec: Rec):
    supported = fp8_matmul_supported(ctx.device)
    rec.metric("fp8_matmul_supported", supported)
    rec.info(f"torch._scaled_mm on {ctx.device}: {'works' if supported else 'not supported'}; "
             f"OneTrainer {'allows' if supported else 'stops before loading with'} float W8A8 / GGUF A8 float")
    if ctx.is_gpu and ctx.is_rocm:
        expected = ctx.arch in FP8_ARCHS_ROCM
        if expected != supported:
            rec.warn(f"expected {'support' if expected else 'no support'} on {ctx.arch}; the probe may be wrong here")
