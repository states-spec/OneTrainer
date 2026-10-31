import contextlib
import importlib.metadata
import os
import sys
from pathlib import Path

from modules.util.rocm_check.framework import Ctx, Rec, check, onetrainer_version
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


def rocm_runtime_lines() -> list[str]:
    """Where the ROCm user-space libraries come from: AMD's pip packages (ROCm 10 builds: rocm-sdk-core) or the ones
    PyTorch's own ROCm wheels bundle, and which HIP runtime this process loaded. Two different HIP runtimes in one
    process (e.g. a library built against /opt/rocm next to the pip one) don't work together."""
    lines = []
    sdk = []
    for name in ("rocm-sdk-core", "rocm-sdk-libraries"):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            sdk.append(f"{name} {importlib.metadata.version(name)}")
    if sdk:
        lines.append("ROCm from pip: " + ", ".join(sdk))
    try:
        with open("/proc/self/maps", encoding="utf-8") as f:
            hip = sorted({line.split()[-1] for line in f if "libamdhip64" in line and "/" in line})
    except OSError:
        hip = []
    if hip:
        lines.append(("HIP runtime loaded from: " if len(hip) == 1 else "SEVERAL HIP runtimes loaded: ") + ", ".join(hip))
    return lines


def _list_gpus(ctx: Ctx, rec: Rec):
    # every visible GPU: an integrated one (a Ryzen CPU's) shows up next to the discrete card on ROCm
    integrated = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        arch = p.gcnArchName.split(":")[0] if ctx.is_rocm else f"sm_{p.major}{p.minor}"
        if getattr(p, "is_integrated", 0):
            integrated.append(i)
        rec.line(f"  cuda:{i}: {p.name}, {arch}, {'integrated' if i in integrated else 'discrete'}, "
                 f"{p.total_memory / 2**30:.1f} GiB" + (" (checked here)" if i == (ctx.device.index or 0) else ""))
    selection = [f"{k}={os.environ[k]}" for k in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
                 if k in os.environ]
    if selection:
        rec.line(f"device selection: {', '.join(selection)}")
    if integrated and torch.cuda.device_count() > 1:
        rec.line("an integrated GPU is visible next to the discrete one. OneTrainer trains on train_device (default "
                 "'cuda' = cuda:0); HIP_VISIBLE_DEVICES set to the discrete GPU's number hides the other one")


@check(SECTION, "PyTorch, GPU, libraries and settings")
def environment(ctx: Ctx, rec: Rec):
    rec.info()
    rec.line(f"OneTrainer {onetrainer_version()}")
    rocm = getattr(torch.version, "rocm", None)  # AMD's own builds (ROCm 10: torch 2.13.0+rocm10.1.0 has HIP 7.16)
    rec.line(f"Python {sys.version.split()[0]}, torch {torch.__version__}, "
             f"HIP {torch.version.hip}{f' (ROCm {rocm})' if rocm else ''}, CUDA {torch.version.cuda}")
    rec.metric("torch", torch.__version__)
    rec.metric("hip", torch.version.hip)
    if rocm:
        rec.metric("rocm", rocm)
    if ctx.is_rocm:
        for line in rocm_runtime_lines():
            (rec.warn if line.startswith("SEVERAL") else rec.line)(line)

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
        _list_gpus(ctx, rec)
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
    except ModuleNotFoundError:
        rec.warn("bitsandbytes is not installed: *_8BIT optimizers, ADAGRAD, RMSPROP, LARS, LAMB, AdEMAMix, INT_8 and "
                 "NFLOAT_4 won't work (requirements-rocm10.txt leaves it out: no release has a build for its ROCm)")
    except Exception as e:
        rec.warn(f"bitsandbytes import failed: {type(e).__name__}: {e}")

    rocm_version = Path("/opt/rocm/.info/version")
    if rocm_version.is_file():
        rec.line(f"system ROCm (/opt/rocm): {rocm_version.read_text(encoding='utf-8').strip()}")
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
