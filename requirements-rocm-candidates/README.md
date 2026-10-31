# ROCm/PyTorch candidates (opt-in, for comparing builds)

Each file here replaces `requirements-rocm.txt` (torch 2.13.0+rocm7.2, the default) with another PyTorch/ROCm build,
to compare speed and stability on your GPU. Install each into its own clone, so your working setup stays as it is:

```sh
git clone -b claude/pyside-6.11.2-rocm-10.1 https://github.com/states-spec/OneTrainer ~/OT_<name>
cd ~/OT_<name>
OT_PLATFORM_REQUIREMENTS=requirements-rocm-candidates/<file> ./install.sh
./run-cmd.sh rocm_check --full
```

Then compare (see docs/GpuCheck.md, "Comparing PyTorch/ROCm builds", and docs/RunDiagnostics.md, `diag_ab` with
`python:` variants, for the real workload). `update.sh` in such a clone needs the same variable, or it installs
`requirements-rocm.txt`. The ROCm 10.1 build is `requirements-rocm10.txt` in the repository root.

| file | torch | ROCm runtime (bundled) | attention (AOTriton) | what it tells |
|---|---|---|---|---|
| (default) `requirements-rocm.txt` | 2.13.0 | 7.2 | 0.12 | the reference (your current setup) |
| `rocm71-torch2.13.txt` | 2.13.0 | 7.1 | 0.12.0 | the ROCm 7.1 libraries (BLAS, MIOpen, HIP) with the same torch and attention |
| `rocm72-torch2.12.txt` | 2.12.1 | 7.2 | 0.11.2 | the older attention kernels; PyTorch keeps gfx11 at head size 256 with 0.11, so head size 512 runs the math kernel there anyway |
| `rocm72-torch2.14.txt` | 2.14.1 | 7.2 | 0.13.0 | the newer torch and attention on the same ROCm |
| (root) `requirements-rocm10.txt` | 2.13.0 | 10.1 (AMD's build) | 0.14 | measured: same results, attention 8-58% slower, no bitsandbytes |

Not here: PyTorch's ROCm 7.14 and 10.0 builds (TheRock 7.14/10.0: the AOTriton packaging gap that wrongly enables
flash attention and gave NaNs), and AMD's ROCm 10.2/10.3 nightlies (no stable release yet).
