"""Correctness of the GPU kernels training runs on, against float64 references computed on the CPU.

Inputs are rounded to the tested dtype before the reference sees them, so the error measured is the kernel's own
(accumulation, output rounding), not the input rounding. Tolerances are relative errors (norm of the difference /
norm of the reference) and leave room for the output dtype's rounding.
"""
import math

from modules.util.rocm_check.framework import Ctx, Rec, check, finite, rel_err
from modules.util.rocm_check.shapes import FAMILIES, HEAD_DIMS

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

SECTION = "2 kernel correctness"

TOL = {torch.float32: 1e-4, torch.float16: 3e-3, torch.bfloat16: 2e-2}
DTYPE_NAME = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}


def _dtypes(ctx: Ctx) -> list[torch.dtype]:
    return [torch.float32, torch.bfloat16, torch.float16] if ctx.full else [torch.float32, torch.bfloat16]


def _rand(ctx: Ctx, *shape, dtype=torch.float32, scale=1.0, seed=None) -> torch.Tensor:
    # made on the CPU with a seeded generator, so the reference and the device see the same numbers
    return (torch.randn(*shape, generator=ctx.generator(seed)) * scale).to(dtype)


@check(SECTION, "matmul / linear forward+backward (BLAS)")
def matmul(ctx: Ctx, rec: Rec):
    shapes = [(333, 3072, 3072), (77, 768, 3072), (1024, 640, 5120), (256, 12288, 3072)]
    if not ctx.full:
        shapes = shapes[:2]
    for dtype in _dtypes(ctx):
        for m, k, n in shapes:
            x = _rand(ctx, m, k, dtype=dtype, seed=1)
            w = _rand(ctx, n, k, dtype=dtype, scale=k ** -0.5, seed=2)
            b = _rand(ctx, n, dtype=dtype, seed=3)
            g = _rand(ctx, m, n, dtype=dtype, seed=4)

            xr, wr, br = (t.double().requires_grad_() for t in (x, w, b))
            F.linear(xr, wr, br).backward(g.double())
            yr = F.linear(xr, wr, br)

            xd, wd, bd = (t.to(ctx.device, copy=True).requires_grad_() for t in (x, w, b))
            y = F.linear(xd, wd, bd)
            y.backward(g.to(ctx.device))
            errs = {"out": rel_err(y, yr), "dx": rel_err(xd.grad, xr.grad), "dw": rel_err(wd.grad, wr.grad),
                    "db": rel_err(bd.grad, br.grad)}
            worst = max(errs.values())
            label = f"{DTYPE_NAME[dtype]} {m}x{k} @ {k}x{n}"
            rec.expect(worst < TOL[dtype] and finite(y), f"{label}: worst rel err {worst:.2e} (limit {TOL[dtype]:.0e})")
            rec.metric(f"linear {label} worst_rel_err", worst)


@check(SECTION, "int8 matmul (torch._int_mm, used by int W8A8)", gpu_only=True)
def int_mm(ctx: Ctx, rec: Rec):
    for m, k, n in [(64, 3072, 3072), (1040, 640, 5120), (333 + 3, 1536, 6144)]:
        m += (-m) % 8
        a = torch.randint(-127, 128, (m, k), generator=ctx.generator(1), dtype=torch.int8)
        w = torch.randint(-127, 128, (n, k), generator=ctx.generator(2), dtype=torch.int8)
        ref = a.long() @ w.long().T
        # the layer calls it with the transposed (column-major) weight, as in LinearW8A8
        out = torch._int_mm(a.to(ctx.device), w.to(ctx.device).T)
        rec.expect(torch.equal(out.cpu().long(), ref), f"{m}x{k} @ {k}x{n} exact")


@check(SECTION, "OneTrainer Triton 8-bit matmul (W8A8 backward)", gpu_only=True)
def triton_mm_8bit(ctx: Ctx, rec: Rec):
    from modules.util.triton_mm_8bit import mm_8bit

    m, k, n = 1024, 3072, 3072
    a = torch.randint(-127, 128, (m, k), generator=ctx.generator(1), dtype=torch.int8)
    w = torch.randint(-127, 128, (k, n), generator=ctx.generator(2), dtype=torch.int8)
    ref = a.long() @ w.long()
    ad, wd = a.to(ctx.device), w.to(ctx.device)
    rec.expect(torch.equal(mm_8bit(ad, wd).cpu().long(), ref), "int8, row-major rhs: exact")
    wt = w.T.contiguous().to(ctx.device).T  # same values, strided like a transposed weight
    rec.expect(torch.equal(mm_8bit(ad, wt).cpu().long(), ref), "int8, strided rhs: exact")

    af = _rand(ctx, m, k, seed=3).clamp(-448, 448).to(torch.float8_e4m3fn)
    wf = _rand(ctx, k, n, seed=4).clamp(-448, 448).to(torch.float8_e4m3fn)
    ref_f = af.double() @ wf.double()
    err = rel_err(mm_8bit(af.to(ctx.device), wf.to(ctx.device)), ref_f)
    rec.expect(err < 1e-4, f"fp8 e4m3: rel err {err:.2e}")

    bf = _rand(ctx, m, k, dtype=torch.bfloat16, seed=5).to(ctx.device)
    wb = _rand(ctx, k, n, dtype=torch.bfloat16, seed=6).to(ctx.device)
    t_bf16 = ctx.bench(lambda: bf @ wb)
    t_int8 = ctx.bench(lambda: mm_8bit(ad, wd))
    t_int8_strided = ctx.bench(lambda: mm_8bit(ad, wt))
    afd, wfd = af.to(ctx.device), wf.to(ctx.device)
    t_fp8 = ctx.bench(lambda: mm_8bit(afd, wfd))
    rec.metric(f"{m}x{k}x{n} ms bf16 matmul", t_bf16)
    rec.metric(f"{m}x{k}x{n} ms triton int8", t_int8)
    rec.metric(f"{m}x{k}x{n} ms triton int8 strided", t_int8_strided)
    rec.metric(f"{m}x{k}x{n} ms triton fp8", t_fp8)
    rec.line(f"int8 is {t_bf16 / t_int8:.2f}x the speed of bf16 (>1 = faster)")


@check(SECTION, "W8A8 linear layers forward+backward", gpu_only=True)
def w8a8_layers(ctx: Ctx, rec: Rec):
    from modules.module.quantized.LinearW8A8 import LinearW8A8
    from modules.util.torch_util import fp8_matmul_supported

    kinds = [("int", torch.int8)]
    if fp8_matmul_supported(ctx.device):
        kinds.append(("float", torch.float8_e4m3fn))
    else:
        rec.line("float W8A8 not tested: this GPU has no fp8 matmul (expected on RDNA3 and older)")
    k, n, m = 3072, 3072, 1024
    base = torch.nn.Linear(k, n, bias=True)
    with torch.no_grad():
        base.weight.copy_(_rand(ctx, n, k, scale=k ** -0.5, seed=1))
        base.bias.copy_(_rand(ctx, n, seed=2))
    x = _rand(ctx, m, k, dtype=torch.bfloat16, seed=3)
    g = _rand(ctx, m, n, dtype=torch.bfloat16, seed=4)

    ref_layer = torch.nn.Linear(k, n).to(ctx.device, torch.bfloat16)
    ref_layer.load_state_dict(base.state_dict())
    ref_layer.requires_grad_(False)
    gd = g.to(ctx.device)
    xr = x.to(ctx.device, copy=True).requires_grad_()
    yr = ref_layer(xr)
    yr.backward(gd)
    ref_y, ref_dx = yr.detach(), xr.grad.clone()  # before timing, which adds to xr.grad
    t_ref = ctx.bench(lambda: ref_layer(xr).backward(gd))

    for name, dtype in kinds:
        layer = LinearW8A8(dtype, k, n, bias=True)
        # filled like quantization_util.__create_linear_layer; load_state_dict would want the "scale" buffer
        layer.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        layer.bias = nn.Parameter(base.bias.detach().to(torch.bfloat16), requires_grad=False)
        layer = layer.to(ctx.device)
        layer.quantize(device=ctx.device)
        layer.compute_dtype = torch.bfloat16
        xd = x.to(ctx.device, copy=True).requires_grad_()
        y = layer(xd)
        y.backward(gd)
        err_y, err_dx = rel_err(y, ref_y), rel_err(xd.grad, ref_dx)
        rec.expect(err_y < 0.05 and err_dx < 0.05 and finite(y) and finite(xd.grad),
                   f"{name} W8A8 vs bf16 linear: out rel err {err_y:.2e}, dx rel err {err_dx:.2e} (limit 5e-2)")
        t = ctx.bench(lambda layer=layer, xd=xd: layer(xd).backward(gd))
        rec.metric(f"{name} W8A8 fwd+bwd ms ({m}x{k}x{n})", t)
        rec.line(f"{name} W8A8 fwd+bwd is {t_ref / t:.2f}x the speed of bf16 ({t:.2f} vs {t_ref:.2f} ms)")
    rec.metric(f"bf16 linear fwd+bwd ms ({m}x{k}x{n})", t_ref)


def _masks(b: int, l_q: int, l_k: int, dtype: torch.dtype, fully_masked_rows: bool):
    # batch 0 has no padding, batch 1 has its last 25% keys padded (like a short caption)
    valid = torch.ones(b, l_k, dtype=torch.bool)
    valid[1, -(l_k // 4):] = False
    masks = {
        "no mask": None,
        "bool key padding (Bx1x1xL)": valid[:, None, None, :],
        f"float key padding ({DTYPE_NAME.get(dtype, dtype)}, -1e4)":
            torch.where(valid, 0.0, -1e4)[:, None, None, :].to(dtype),
    }
    if fully_masked_rows and l_q == l_k:
        # the query x key outer product (Chroma's mask): padded query rows have no valid key at all
        masks["bool LxL outer product"] = valid[:, None, None, :] & valid[:, None, :, None]
    return masks


def _attention_reference(q, k, v, mask):
    q, k, v = (t.double() for t in (q, k, v))
    scores = q @ k.transpose(-1, -2) * q.shape[-1] ** -0.5
    if mask is None:
        return torch.softmax(scores, dim=-1) @ v
    if mask.dtype != torch.bool:
        return torch.softmax(scores + mask.double(), dim=-1) @ v
    # a finite fill keeps the gradient finite; rows without any valid key give 0 (output and gradients), like
    # torch's math kernel
    p = torch.softmax(scores.masked_fill(~mask, -1e30), dim=-1) * mask.any(dim=-1, keepdim=True)
    return p @ v


@check(SECTION, "SDPA backend chosen per head size, dtype and mask")
def sdpa_choice(ctx: Ctx, rec: Rec):
    """which kernel scaled_dot_product_attention picks by itself: MATH is the slow, memory-hungry fallback"""
    rec.info()
    b, seq = 2, 1024
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        for d in HEAD_DIMS + [256]:
            q = torch.zeros(b, 4, seq, d, dtype=dtype, device=ctx.device)
            masks = _masks(b, seq, seq, dtype, fully_masked_rows=True)
            masks["float32 key padding (dtype differs from q)"] = \
                None if dtype == torch.float32 else torch.zeros(b, 1, 1, seq, device=ctx.device)
            row = []
            for name, mask in masks.items():
                if mask is None and name != "no mask":
                    continue
                if mask is not None:
                    mask = mask.to(ctx.device)
                choice = SDPBackend(torch._fused_sdp_choice(q, q, q, attn_mask=mask))
                row.append(f"{name}: {choice.name}")
                if ctx.is_gpu and choice == SDPBackend.MATH and "differs" not in name and dtype != torch.float32:
                    rec.warn(f"head dim {d} {DTYPE_NAME[dtype]} {name}: falls back to the MATH kernel")
            rec.line(f"head dim {d} {DTYPE_NAME[dtype]}: " + "; ".join(row))
    for family in FAMILIES:
        for res in ctx.resolutions:
            s = family.seq(res)
            q = torch.zeros(1, family.heads, s, family.head_dim, dtype=torch.bfloat16, device=ctx.device)
            choice = SDPBackend(torch._fused_sdp_choice(q, q, q))
            rec.line(f"{family.name} @ {res}px ({family.heads}x{family.head_dim}, seq {s}), bf16: {choice.name}")


@check(SECTION, "SDPA forward+backward per backend, head size and mask")
def sdpa_correctness(ctx: Ctx, rec: Rec):
    b, h, seq = 2, 3, 257  # odd length: exercises the kernels' tail handling
    backends = [None, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    dtypes = [torch.bfloat16, torch.float16] if ctx.full else [torch.bfloat16]
    for dtype in dtypes:
        for d in HEAD_DIMS:
            q, k, v = (_rand(ctx, b, h, seq, d, dtype=dtype, seed=s) for s in (1, 2, 3))
            g = _rand(ctx, b, h, seq, d, dtype=dtype, seed=4)
            for mask_name, mask in _masks(b, seq, seq, dtype, fully_masked_rows=True).items():
                qr, kr, vr = (t.double().requires_grad_() for t in (q, k, v))
                ref = _attention_reference(qr, kr, vr, mask)
                ref.backward(g.double())
                rows_valid = torch.ones(b, 1, seq, 1, dtype=torch.bool) if mask is None or mask.shape[-2] == 1 \
                    else mask.any(dim=-1, keepdim=True)
                results = []
                for backend in backends:
                    bname = "default" if backend is None else backend.name.replace("_ATTENTION", "")
                    qd, kd, vd = (t.to(ctx.device, copy=True).requires_grad_() for t in (q, k, v))
                    md = None if mask is None else mask.to(ctx.device)
                    try:
                        if backend is None:
                            out = F.scaled_dot_product_attention(qd, kd, vd, attn_mask=md)
                        else:
                            with sdpa_kernel([backend]):
                                out = F.scaled_dot_product_attention(qd, kd, vd, attn_mask=md)
                        out.backward(g.to(ctx.device))
                    except RuntimeError as e:
                        if any(s in str(e) for s in ("No available kernel", "No viable backend")) \
                                or "not supported" in str(e).lower():
                            results.append(f"{bname} n/a")
                            continue
                        raise
                    valid = rows_valid.expand_as(out).to(ctx.device)
                    err_out = rel_err(out[valid], ref[valid.cpu()])
                    err_grad = max(rel_err(x.grad, r.grad) for x, r in ((qd, qr), (kd, kr), (vd, vr)))
                    all_finite = finite(out) and all(finite(x.grad) for x in (qd, kd, vd))
                    label = f"hd {d} {DTYPE_NAME[dtype]} {mask_name} {bname}"
                    tol = TOL[dtype] * 2
                    if not math.isfinite(err_out) or not math.isfinite(err_grad):
                        rec.fail(f"{label}: the comparison itself is not finite ({err_out}, {err_grad})")
                    elif not all_finite:
                        rec.fail(f"{label}: NaN/inf in the output or gradients"
                                 f"{' (fully masked query rows)' if 'outer' in mask_name else ''}")
                    elif err_out > tol or err_grad > tol * 2.5:
                        rec.fail(f"{label}: rel err out {err_out:.2e}, grads {err_grad:.2e}")
                    results.append(f"{bname} {err_out:.1e}/{err_grad:.1e}")
                rec.line(f"hd {d} {DTYPE_NAME[dtype]} {mask_name}: " + ", ".join(results))
    rec.line("(numbers: rel err of output / worst gradient; n/a = the backend doesn't take this input)")


@check(SECTION, "convolution, group norm, upsample forward+backward (VAE ops)")
def conv_ops(ctx: Ctx, rec: Rec):
    for dtype in _dtypes(ctx):
        x = _rand(ctx, 2, 128, 64, 64, dtype=dtype, seed=1)
        w = _rand(ctx, 256, 128, 3, 3, dtype=dtype, scale=(128 * 9) ** -0.5, seed=2)
        gn_w = _rand(ctx, 128, dtype=dtype, seed=3)

        def ops(x, w, gn_w):
            h = F.group_norm(x, 32, gn_w)
            h = F.silu(h)
            h = F.interpolate(h, scale_factor=2.0, mode="nearest")
            return F.conv2d(h, w, padding=1)

        xr, wr, gr = (t.double().requires_grad_() for t in (x, w, gn_w))
        out_r = ops(xr, wr, gr)
        g = _rand(ctx, *out_r.shape, dtype=dtype, seed=4)
        out_r.backward(g.double())
        xd, wd, gd = (t.to(ctx.device, copy=True).requires_grad_() for t in (x, w, gn_w))
        out = ops(xd, wd, gd)
        out.backward(g.to(ctx.device))
        errs = {"out": rel_err(out, out_r), "dx": rel_err(xd.grad, xr.grad), "dw": rel_err(wd.grad, wr.grad),
                "dgn": rel_err(gd.grad, gr.grad)}
        worst = max(errs.values())
        tol = TOL[dtype] * (10 if dtype == torch.float32 else 2)  # MIOpen may use Winograd for fp32
        rec.expect(worst < tol and finite(out),
                   f"{DTYPE_NAME[dtype]}: " + ", ".join(f"{k} {v:.1e}" for k, v in errs.items()) + f" (limit {tol:.0e})")


@check(SECTION, "bitsandbytes (8-bit optimizer state, INT_8, NFLOAT_4)", gpu_only=True)
def bitsandbytes_ops(ctx: Ctx, rec: Rec):
    from modules.module.quantized.LinearNf4 import LinearNf4

    import bitsandbytes as bnb
    import bitsandbytes.cextension as bnb_ext

    if isinstance(bnb_ext.lib, bnb_ext.ErrorHandlerMockBNBNativeLibrary):
        rec.fail("the native library failed to load (see its error at startup); *_8BIT optimizers, INT_8 and "
                 "NFLOAT_4 won't work")
        return

    k, n = 1024, 1024
    base = torch.nn.Linear(k, n)
    with torch.no_grad():
        base.weight.copy_(_rand(ctx, n, k, scale=k ** -0.5, seed=1))
    x = _rand(ctx, 64, k, dtype=torch.bfloat16, seed=2).to(ctx.device)
    ref = F.linear(x.float(), base.weight.to(ctx.device), base.bias.to(ctx.device))

    def fill(layer):  # as quantization_util.__create_linear_layer fills the quantized layers
        layer.weight = type(layer.weight)(base.weight.detach().clone(), requires_grad=False)
        layer.bias = type(layer.bias)(base.bias.detach().clone(), requires_grad=False)
        return layer

    nf4 = fill(LinearNf4(k, n, bias=True))
    nf4.compute_dtype = torch.bfloat16
    nf4.quantize(device=ctx.device)
    nf4 = nf4.to(ctx.device)
    with torch.autocast(ctx.device.type, dtype=torch.bfloat16):  # as in training
        err = rel_err(nf4(x), ref)
    rec.expect(err < 0.15, f"NFLOAT_4 linear (OneTrainer LinearNf4) vs fp32: rel err {err:.3f} (limit 0.15)")

    int8 = fill(bnb.nn.Linear8bitLt(k, n, bias=True, has_fp16_weights=False)).to(ctx.device)  # quantizes here
    with torch.autocast(ctx.device.type, dtype=torch.bfloat16):
        err = rel_err(int8(x), ref)
    rec.expect(err < 0.05, f"INT_8 linear (bnb Linear8bitLt) vs fp32: rel err {err:.3f} (limit 0.05)")

    # 8-bit AdamW state vs torch's AdamW over 10 steps: compare how far each moved the weights
    p0 = _rand(ctx, 256, 1024, seed=3)
    moved = {}
    for name, cls in (("torch", torch.optim.AdamW), ("bnb 8-bit", bnb.optim.AdamW8bit)):
        p = torch.nn.Parameter(p0.clone().to(ctx.device))
        opt = cls([p], lr=1e-3)
        for step in range(10):
            p.grad = _rand(ctx, 256, 1024, seed=100 + step).to(ctx.device)
            opt.step()
        moved[name] = p.detach() - p0.to(ctx.device)
        rec.expect(finite(p), f"AdamW ({name}) weights finite")
    err = rel_err(moved["bnb 8-bit"], moved["torch"])
    rec.expect(err < 0.1, f"AdamW8bit vs AdamW update: rel err {err:.3f} (limit 0.1)")


@check(SECTION, "bf16 stochastic rounding")
def stochastic_rounding(ctx: Ctx, rec: Rec):
    from modules.util.bf16_stochastic_rounding import copy_stochastic_, set_seed

    set_seed(0, ctx.device)
    lo = torch.tensor(1.0, dtype=torch.bfloat16)
    hi = (lo.float() + 2 ** -7).bfloat16()  # the next bf16 value above 1.0
    value = lo.float() + 0.3 * (hi.float() - lo.float())
    source = torch.full((1_000_000,), value.item(), device=ctx.device)
    target = torch.empty_like(source, dtype=torch.bfloat16)
    copy_stochastic_(target, source)
    values = set(torch.unique(target.float()).cpu().tolist())
    frac_up = (target == hi.to(ctx.device)).float().mean().item()
    rec.expect(values <= {lo.item(), hi.item()}, f"rounds only to the two neighbours ({sorted(values)})")
    rec.expect(abs(frac_up - 0.3) < 0.01, f"rounds up {frac_up:.3f} of the time (expected 0.300)")
