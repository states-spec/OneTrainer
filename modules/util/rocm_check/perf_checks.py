"""Speed of the kernels training spends its time in, per model family and resolution, and what the ROCm runtime
switches (BLAS library, TunableOp, MIOpen find mode, allocator settings, torch.compile) change. Numbers only: a slow
result is reported, not failed, except where a setting breaks.
"""
import os
import time

from modules.util.rocm_check.framework import Ctx, Rec, check, child_task, finite, rel_err, run_child
from modules.util.rocm_check.shapes import FAMILIES

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

SECTION = "4 speed"


def _rand(ctx: Ctx, *shape, dtype=torch.bfloat16, scale=1.0, seed=0) -> torch.Tensor:
    g = torch.Generator(device=ctx.device).manual_seed(seed)
    return (torch.randn(*shape, generator=g, device=ctx.device) * scale).to(dtype)


def _tflops(flops: float, ms: float) -> float:
    return flops / ms / 1e9


@check(SECTION, "attention fwd+bwd per model family and backend", gpu_only=True)
def attention_speed(ctx: Ctx, rec: Rec):
    rec.info()
    backends = [None, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
    if ctx.full:
        backends.append(SDPBackend.MATH)
    for family in FAMILIES:
        for res in ctx.resolutions:
            s = family.seq(res)
            q, k, v = (_rand(ctx, 1, family.heads, s, family.head_dim, seed=i).requires_grad_() for i in range(3))
            flops = 4 * family.heads * s * s * family.head_dim * 3.5  # fwd (2 matmuls) + bwd (~2.5x)
            parts = []
            for backend in backends:
                name = "default" if backend is None else backend.name.replace("_ATTENTION", "").lower()
                if backend == SDPBackend.MATH and s > 4608:
                    continue  # the math kernel materializes seq x seq scores: skip what can't fit
                torch.cuda.reset_peak_memory_stats(ctx.device)
                before = torch.cuda.memory_allocated(ctx.device)

                def run(backend=backend, q=q, k=k, v=v):
                    if backend is None:
                        out = F.scaled_dot_product_attention(q, k, v)
                    else:
                        with sdpa_kernel([backend]):
                            out = F.scaled_dot_product_attention(q, k, v)
                    out.backward(torch.ones_like(out))

                try:
                    ms = ctx.bench(run, iters=10 if ctx.full else 4, warmup=2)
                except RuntimeError as e:
                    if any(s in str(e) for s in ("No available kernel", "No viable backend")):
                        parts.append(f"{name} n/a")
                        continue
                    raise
                except torch.OutOfMemoryError:
                    parts.append(f"{name} OOM")
                    torch.cuda.empty_cache()
                    continue
                peak = (torch.cuda.max_memory_allocated(ctx.device) - before) / 2**20
                parts.append(f"{name} {ms:.2f} ms ({_tflops(flops, ms):.0f} TFLOPS, +{peak:.0f} MiB)")
                rec.metric(f"{family.name} @{res} {name} ms", ms)
            rec.line(f"{family.name} @ {res}px (1x{family.heads}x{s}x{family.head_dim} bf16): " + "; ".join(parts))
            del q, k, v


@check(SECTION, "linear layers fwd+bwd per model family (bf16)", gpu_only=True)
def linear_speed(ctx: Ctx, rec: Rec):
    rec.info()
    for family in FAMILIES:
        res = ctx.resolutions[0]
        tokens = family.seq(res)
        x = _rand(ctx, tokens, family.hidden, seed=1).requires_grad_()
        results = []
        for in_f, out_f in ((family.hidden, family.hidden * 3), (family.hidden, family.mlp), (family.mlp, family.hidden)):
            w = _rand(ctx, out_f, in_f, scale=in_f ** -0.5, seed=2)  # frozen base weight, like LoRA training
            xi = x if in_f == family.hidden else _rand(ctx, tokens, in_f, seed=3).requires_grad_()
            grad = torch.ones(tokens, out_f, device=ctx.device, dtype=torch.bfloat16)
            ms = ctx.bench(lambda xi=xi, w=w, grad=grad: F.linear(xi, w).backward(grad))
            tf = _tflops(2 * tokens * in_f * out_f * 2, ms)  # forward + input gradient
            results.append(f"{in_f}->{out_f} {ms:.2f} ms ({tf:.0f} TFLOPS)")
            rec.metric(f"{family.name} {tokens}x{in_f}->{out_f} TFLOPS", tf)
        rec.line(f"{family.name} @ {res}px, {tokens} tokens: " + "; ".join(results))


@check(SECTION, "VAE convolutions (MIOpen) per resolution", gpu_only=True)
def conv_speed(ctx: Ctx, rec: Rec):
    rec.info()
    for dtype in (torch.float32, torch.bfloat16):
        for res in ctx.resolutions:
            x = _rand(ctx, 1, 128, res, res, dtype=dtype, seed=1)
            w = _rand(ctx, 128, 128, 3, 3, dtype=dtype, scale=(128 * 9) ** -0.5, seed=2)
            t0 = time.perf_counter()
            F.conv2d(x, w, padding=1)
            ctx.sync()
            first = (time.perf_counter() - t0) * 1000
            ms = ctx.bench(lambda x=x, w=w: F.conv2d(x, w, padding=1))
            name = str(dtype)[6:]
            rec.line(f"{name} 128ch 3x3 at {res}x{res}: {ms:.2f} ms (first call {first:.0f} ms incl. kernel search)")
            rec.metric(f"conv {name} {res}px ms", ms)


@child_task("core_bench")
def core_bench(ctx: Ctx) -> dict:
    """a fixed mix of what training runs, timed in a fresh process (so its environment variables apply)"""
    out = {}
    t0 = time.perf_counter()
    for name, (m, k, n) in {"linear 1536x3072x12288": (1536, 3072, 12288),
                            "linear 4096x640x5120": (4096, 640, 5120),
                            "linear 1357x1536x6144": (1357, 1536, 6144)}.items():
        x = _rand(ctx, m, k, seed=1).requires_grad_()
        w = _rand(ctx, n, k, scale=k ** -0.5, seed=2)
        g = torch.ones(m, n, device=ctx.device, dtype=torch.bfloat16)
        out[name + " ms"] = ctx.bench(lambda x=x, w=w, g=g: F.linear(x, w).backward(g), iters=10 if ctx.full else 5)
    q = _rand(ctx, 1, 24, 1536, 128, seed=3).requires_grad_()
    out["attention 24x1536x128 ms"] = ctx.bench(
        lambda: F.scaled_dot_product_attention(q, q, q).backward(torch.ones_like(q)), iters=10 if ctx.full else 5)
    x = _rand(ctx, 1, 128, 512, 512, dtype=torch.float32, seed=4)
    w = _rand(ctx, 128, 128, 3, 3, dtype=torch.float32, scale=0.03, seed=5)
    t1 = time.perf_counter()
    F.conv2d(x, w, padding=1)
    ctx.sync()
    out["conv first call ms"] = (time.perf_counter() - t1) * 1000
    out["conv fp32 512px ms"] = ctx.bench(lambda: F.conv2d(x, w, padding=1))
    out["total s"] = time.perf_counter() - t0
    return out


@check(SECTION, "runtime switches: BLAS library, TunableOp, MIOpen find mode", gpu_only=True, timeout=1800)
def runtime_switches(ctx: Ctx, rec: Rec):
    rec.info()
    variants = {"default": {}}
    if ctx.is_rocm:
        variants["rocBLAS instead of hipBLASLt (TORCH_BLAS_PREFER_HIPBLASLT=0)"] = {"TORCH_BLAS_PREFER_HIPBLASLT": "0"}
        variants["MIOPEN_FIND_MODE=FAST"] = {"MIOPEN_FIND_MODE": "FAST"}
    if ctx.full:
        variants["TunableOp (PYTORCH_TUNABLEOP_ENABLED=1, tuned in this run)"] = {
            "PYTORCH_TUNABLEOP_ENABLED": "1", "PYTORCH_TUNABLEOP_TUNING": "1",
            "PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS": "100",
            "PYTORCH_TUNABLEOP_FILENAME": os.path.join(ctx.out_dir, "rocm_check_tunableop%d.csv"),
        }
    if ctx.is_rocm:
        # each run gets an empty MIOpen tuning database, so "conv first call" compares like with like
        for i, env in enumerate(variants.values()):
            db = os.path.join(ctx.out_dir, "rocm_check_miopen_db", str(i))
            os.makedirs(db, exist_ok=True)
            env["MIOPEN_USER_DB_PATH"] = db
    results = {name: run_child(ctx, "core_bench", env) for name, env in variants.items()}
    base = results["default"]
    if "error" in base:
        rec.fail(f"default run failed: {base['error']}")
        return
    for name, result in results.items():
        if "error" in result:
            rec.warn(f"{name}: {result['error']}")
            continue
        parts = []
        for key, value in result.items():
            if key.endswith(" ms") and key in base:
                ratio = base[key] / value if value else float("nan")
                parts.append(f"{key.removesuffix(' ms')} {value:.2f} ms" + ("" if name == "default" else f" ({ratio:.2f}x)"))
                rec.metric(f"{name}: {key}", value)
        rec.line(f"{name} (total {result.get('total s', 0):.0f} s): " + "; ".join(parts))
    rec.line("(x = speed relative to default; >1 is faster. TunableOp's first run includes tuning; its results are "
             "saved in rocm_check_tunableop*.csv)")


@child_task("alloc_pattern")
def alloc_pattern(ctx: Ctx) -> dict:
    """allocates and frees tensors of changing sizes, like steps over aspect ratio buckets and model swaps"""
    free, _ = torch.cuda.mem_get_info(ctx.device)
    torch.cuda.reset_peak_memory_stats(ctx.device)
    kept = []
    for round_ in range(6):
        # sizes relative to the free memory, growing each round: at most ~45% of it is held at once
        for fraction in (0.04, 0.008, 0.08, 0.004, 0.06, 0.02, 0.12, 0.003):
            kept.append(torch.empty(int(free * fraction * (1 + 0.1 * round_)) // 2, dtype=torch.bfloat16,
                                    device=ctx.device))
            if len(kept) > 4:
                kept.pop(0)
    kept.clear()
    big = torch.empty(int(free * 0.4) // 2, dtype=torch.bfloat16, device=ctx.device)  # a model part after the churn
    del big
    return {"peak allocated GiB": torch.cuda.max_memory_allocated(ctx.device) / 2**30,
            "peak reserved GiB": torch.cuda.max_memory_reserved(ctx.device) / 2**30,
            "alloc conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")}


@check(SECTION, "allocator: default vs expandable_segments (OT_EXPANDABLE_SEGMENTS)", gpu_only=True)
def allocator(ctx: Ctx, rec: Rec):
    rec.info()
    existing = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    expandable = ",".join(filter(None, [existing, "expandable_segments:True"]))
    for name, conf in (("default", existing), ("expandable_segments", expandable)):
        result = run_child(ctx, "alloc_pattern", {"PYTORCH_CUDA_ALLOC_CONF": conf})
        if "error" in result:
            (rec.fail if name == "default" else rec.warn)(f"{name}: {result['error']}")
            continue
        waste = result["peak reserved GiB"] - result["peak allocated GiB"]
        rec.line(f"{name} ({conf or 'no settings'}): peak {result['peak allocated GiB']:.2f} GiB in tensors, "
                 f"{result['peak reserved GiB']:.2f} GiB reserved ({waste:.2f} GiB held unused)")
        rec.metric(f"{name} reserved-unused GiB", waste)


class _DiTBlock(nn.Module):
    """a generic pre-norm transformer block (attention + MLP), the shape most of the DiT models share"""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.out = nn.Linear(dim, dim * 3), nn.Linear(dim, dim)
        self.mlp1, self.mlp2 = nn.Linear(dim, dim * 4), nn.Linear(dim * 4, dim)

    def forward(self, x):
        b, s, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, s, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, s, d)
        x = x + self.out(a)
        return x + self.mlp2(F.gelu(self.mlp1(self.norm2(x)), approximate="tanh"))


@check(SECTION, "torch.compile on a transformer block (the 'compile' option)", gpu_only=True, full_only=True,
       timeout=1800)
def compile_speed(ctx: Ctx, rec: Rec):
    rec.info()
    from modules.util.compile_util import init_compile

    init_compile()
    torch.manual_seed(0)
    block = _DiTBlock(1536, 24).to(ctx.device, torch.bfloat16).requires_grad_(False)
    x = _rand(ctx, 1, 1357, 1536, seed=1).requires_grad_()
    ref = block(x)
    eager_ms = ctx.bench(lambda: block(x).float().sum().backward())
    from torch._dynamo.utils import counters

    counters.clear()
    compiled = torch.compile(block, fullgraph=True)
    t0 = time.perf_counter()
    out = compiled(x)
    out.float().sum().backward()
    ctx.sync()
    compile_s = time.perf_counter() - t0
    graphs = counters["stats"].get("unique_graphs", 0)
    cache_hits = sum(v for k, v in counters["inductor"].items() if "cache_hit" in k)
    rec.line(f"dynamo compiled {graphs} graph(s), {sum(counters['graph_break'].values())} graph break(s), "
             f"{cache_hits} inductor cache hit(s) (a hit loads kernels compiled by an earlier run)")
    if graphs == 0:
        rec.warn("torch.compile compiled nothing: the timing below is eager")
    err = rel_err(out, ref)
    rec.expect(finite(out) and err < 3e-2, f"compiled output vs eager: rel err {err:.1e}")
    compiled_ms = ctx.bench(lambda: compiled(x).float().sum().backward())
    rec.line(f"eager {eager_ms:.2f} ms, compiled {compiled_ms:.2f} ms ({eager_ms / compiled_ms:.2f}x), "
             f"first compile {compile_s:.0f} s")
    rec.metric("eager ms", eager_ms)
    rec.metric("compiled ms", compiled_ms)
    rec.metric("compile seconds", compile_s)

