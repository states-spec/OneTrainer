# GPU check (`scripts/rocm_check.py`)

Checks and measures what OneTrainer runs on your GPU, without a model download or a dataset. It was made for AMD
GPUs on ROCm, where kernels are less tested than on CUDA, but it runs on CUDA too, and partly on the CPU.

```sh
./run-cmd.sh rocm_check            # quick mode, about 5 minutes
./run-cmd.sh rocm_check --full     # more dtypes, sizes and optimizers, plus TunableOp and torch.compile (30-45 min)
./run-cmd.sh rocm_check --list     # the checks
./run-cmd.sh rocm_check --only sdpa offloading     # checks whose "section / name" contains one of these
./run-cmd.sh rocm_check --resolutions 512 768 1024 # sizes for the speed checks (default 512 1024)
```

It writes `rocm_check_report.txt` (for reading) and `rocm_check_report.json` (for comparing runs) to the current
directory, or to `--out-dir`. The files are rewritten after every check, so a crash or a GPU hang still leaves the
results up to the check that was running (the report names it). The exit code is 1 if any check failed.

## What it checks

1. **Environment**: PyTorch and its ROCm/CUDA build, the GPU (name, gfx arch, VRAM), the BLAS and flash attention
   libraries, Triton, bitsandbytes, package versions and the GPU environment variables. It warns when AOTriton
   treats the GPU as experimental (then SDPA silently uses its slow math kernel), and reports whether the float W8A8
   weight types can run.
2. **Kernel correctness**, each against a float64 reference on the CPU:
   - matmul/linear forward and backward in fp32/bf16/fp16 (hipBLASLt or cuBLAS)
   - int8 matmul (`torch._int_mm`) and OneTrainer's Triton 8-bit kernel (the W8A8 backward), W8A8 layers
   - scaled dot product attention per backend (flash, memory-efficient, math, and the one PyTorch picks), per head
     size used by the model families, with no mask, key-padding masks and the query x key mask whose padded rows
     have no valid key (NaN there would spread through the model)
   - which attention backend PyTorch picks per head size, dtype and mask, and per model family
   - convolution, group norm and upsampling (the VAE's ops, MIOpen on ROCm)
   - bitsandbytes: NFLOAT_4 and INT_8 layers, 8-bit AdamW
   - bf16 stochastic rounding
3. **Training code** on the GPU:
   - LoRA, DoRA, LoKr (also with DoRA and the vec trick), LoHa and OFT v2, forward and backward under bf16 autocast
   - optimizers built by OneTrainer's own `create_optimizer` (ADV optimizers with every state precision, AdamW,
     schedule-free ones), with a save/resume round trip like a backup
   - layer and activation offloading through the real `LayerOffloadConductor`: same loss and gradients as without
     offloading, weights bit-identical after load/unload cycles
   - pinned host memory and transfer speed
   - concurrent VAE encodes, as caching with `dataloader_threads` > 1 does, with and without the lock OneTrainer
     now uses (on ROCm, unlocked concurrent encodes once gave random NaN latents)
4. **Speed**: attention (per backend) and linear layers per model family and resolution, VAE convolutions, and the
   runtime switches that can only be set at startup, each in a fresh process: rocBLAS instead of hipBLASLt,
   TunableOp, MIOpen's find mode, the allocator with `expandable_segments`, and `torch.compile`.

## Reading the results

- `PASS`: correct within the tolerance shown. `FAIL`: wrong result, NaN/inf or an error. `WARN`: works, but
  something is off (a slow fallback, concurrent encodes breaking). `INFO`: measurements only. `SKIP`: needs a GPU or
  `--full`.
- Correctness numbers are relative errors: the norm of the difference divided by the norm of the reference. bf16
  results land around 1e-3 to 1e-2, fp32 around 1e-7.
- Speed ratios are relative to the default settings; above 1 is faster.

## Adding to it

- A new model family with other attention or MLP sizes: add a row to `modules/util/rocm_check/shapes.py`.
- A new check: a function `(ctx, rec)` with the `@check(section, name)` decorator in one of the modules under
  `modules/util/rocm_check/`. `rec.expect(ok, text)` records a pass/fail line, `rec.warn`/`rec.info`/`rec.metric`
  the rest; exceptions and out-of-memory errors are caught and reported as failures. A setting that PyTorch or ROCm
  reads only at startup needs a `@child_task` and `run_child(ctx, task, env)`.
