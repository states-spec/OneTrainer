# ROCm/PyTorch version comparison (2026-10-31)

**Result: keep the default, torch 2.13.0+rocm7.2 (`requirements-rocm.txt`).** On an RX 7900 XTX (gfx1100) no other
build was faster where training spends its time, and none was more correct. The opt-in requirement files used for the
comparison were removed afterwards; their exact pins are below, for a future comparison.

## What was compared

One `rocm_check --full` per build, each in its own clone and venv (Fedora 44, kernel 7.2.13, Python 3.13, same
OneTrainer code apart from the check fixes listed below):

| Build | torch | ROCm runtime | Attention (AOTriton) | Attention speed vs default | Correctness | bitsandbytes |
|---|---|---|---|---|---|---|
| **default** | 2.13.0+rocm7.2 | 7.2 (PyTorch's build) | 0.12 | 1.00x | all correct | yes |
| torch 2.14 | 2.14.1+rocm7.2 | 7.2 (PyTorch's build) | 0.13.0 | 0.98x overall, Chroma 512 px 0.89x (one run) | all correct | yes |
| ROCm 10.1 | 2.13.0+rocm10.1.0 | 10.1 (AMD's build, HIP 7.16) | 0.14 | 0.83x | all correct | no build exists |
| torch 2.12 | 2.12.1+rocm7.2 | 7.2 (PyTorch's build) | 0.11.2 | 0.57x | all correct | yes |

- **Correctness:** every correctness check, all tiny Chroma training variants (bit-identical to their baseline) and
  every training feature passed in all four; the optimizer and training-feature results were identical to the digit.
- **Speed:** only the attention kernels differ. Linear layers, VAE convolutions, offloading, transfers and adapters were
  within a few percent everywhere (ROCm 10.1's MIOpen: bf16 convs 1.4-1.5x faster, fp32 at 1024 px 1.6x slower).
- **Head size 512 (every VAE's mid-block attention):** PyTorch's flash/efficient kernels give wrong numbers on AOTriton
  0.12/0.13, raise an error on 0.14 (no gfx110x kernels above 256: "profiler is not initialized"), and aren't used on
  0.11 (PyTorch keeps gfx11 at 256 there). OneTrainer's `rocm_sdpa_fix` sends that head size to the math kernel on every
  build, so training is correct on all of them.
- **`expandable_segments`:** wrong results on every build (offloading, VAE encodes, concurrent encodes; which check
  breaks varies between runs). Keep `OT_EXPANDABLE_SEGMENTS` off.
- **First run in a new environment:** MIOpen's find database and Triton's autotuning start empty (first bf16 conv up to
  2 s, the W8A8 checks 30-45 s instead of 3-4 s); that is one-time cost, not a speed difference.

Not tried: PyTorch's ROCm 7.14 and 10.0 builds (TheRock 7.14/10.0: an AOTriton packaging gap that wrongly enables flash
attention and gave NaNs), AMD's ROCm 10.2/10.3 nightlies (no stable release then), and torch 2.13.0+rocm7.1 (the
default's torch and AOTriton with ROCm 7.1's libraries: nothing to gain).

## Doing it again

Tools that stay in the repository:

- `rocm_check --full` in each environment; its report header names the OneTrainer commit and the PyTorch build.
- `./run-cmd.sh rocm_check_compare a.json b.json ...`: reports side by side, statuses and every speed number relative to
  the first report (docs/GpuCheck.md, "Comparing PyTorch/ROCm builds").
- `./run-cmd.sh diag_ab --config-path cfg.json --variant 'a python:~/OT_A/venv/bin/python' --variant 'b python:~/OT_B/venv/bin/python'`:
  the real workload in two environments, alternating (docs/RunDiagnostics.md).

Install a candidate into a separate clone with a requirements file passed through `OT_PLATFORM_REQUIREMENTS` (and the
same variable for `update.sh` there, or it goes back to `requirements-rocm.txt`). The files used, to recreate:

```
# torch 2.14.1 on ROCm 7.2 (PyTorch's build)
--extra-index-url https://download.pytorch.org/whl/rocm7.2
torch==2.14.1+rocm7.2; sys_platform == "linux"
torchvision==0.29.1+rocm7.2; sys_platform == "linux"
onnxruntime==1.23.2
bitsandbytes==0.49.2; sys_platform == "linux"

# torch 2.12.1 on ROCm 7.2 (PyTorch's build)
--extra-index-url https://download.pytorch.org/whl/rocm7.2
torch==2.12.1+rocm7.2; sys_platform == "linux"
torchvision==0.27.1+rocm7.2; sys_platform == "linux"
onnxruntime==1.23.2
bitsandbytes==0.49.2; sys_platform == "linux"

# torch 2.13.0 on ROCm 10.1 (AMD's builds, gfx1100 packages). Pin every AMD package exactly: PyPI has placeholder
# packages under the same names (rocm, rocm-sdk-*, amd-torch-device-*), and torch only asks for rocm[libraries]~=10.1.0.
# amd-torch-device-gfx110x holds the AOTriton (flash attention) kernels; without it torch thinks flash is there while its
# kernels aren't. No bitsandbytes: its newest release (0.50.2) has ROCm builds up to rocm715, none for HIP 7.16.
--extra-index-url https://stable.repo.amd.com/rocm/whl-next/
torch[device-gfx1100]==2.13.0+rocm10.1.0; sys_platform == "linux"
torchvision[device-gfx1100]==0.28.0+rocm10.1.0; sys_platform == "linux"
amd-torch-device-gfx1100==2.13.0+rocm10.1.0; sys_platform == "linux"
amd-torch-device-gfx110x==2.13.0+rocm10.1.0; sys_platform == "linux"
amd-torchvision-device-gfx1100==0.28.0+rocm10.1.0; sys_platform == "linux"
triton==3.8.0+git669b31ac.rocm10.1.0; sys_platform == "linux"
rocm[libraries]==10.1.0; sys_platform == "linux"
rocm-sdk-core==10.1.0; sys_platform == "linux"
rocm-sdk-libraries==10.1.0; sys_platform == "linux"
rocm-sdk-device-gfx1100==10.1.0; sys_platform == "linux"
rocm-bootstrap==0.3.0; sys_platform == "linux"
onnxruntime==1.23.2
```

What to look at first in a new build: its AOTriton version (`torch/lib/libaotriton_v2.so.*` in the wheel) and whether
its gfx110x images have head sizes above 256, the attention row of `rocm_check_compare`, and the SDPA and VAE-per-layer
checks (head size 512 must stay a WARN, never wrong numbers in OneTrainer's path).
