"""The training-side code on the GPU: adapters, optimizers (through OneTrainer's own create_optimizer, with a
save/resume round trip), layer offloading through the real LayerOffloadConductor, pinned memory, and concurrent
VAE encodes (the ROCm NaN-latent bug the caching lock works around).
"""
import copy
import io
import os
import threading
import time

from modules.util import rocm_sdpa_fix
from modules.util.rocm_check.framework import Ctx, Rec, check, finite, rel_err

import torch
import torch.nn.functional as F
from torch import nn

SECTION = "3 training code"


def _rand(ctx: Ctx, *shape, scale=1.0, seed=None) -> torch.Tensor:
    return torch.randn(*shape, generator=ctx.generator(seed)) * scale


def _adapter_factories(device: torch.device):
    from modules.module.LoRAModule import DoRAModule, LoHaModule, LoKrModule, LoRAModule, OFTModule

    return {
        "LoRA": lambda lin, dev: LoRAModule("p", lin, 16, 1.0),
        "DoRA": lambda lin, dev: DoRAModule("p", lin, 16, 1.0, norm_epsilon=False, decompose_output_axis=False,
                                            train_device=dev),
        "LoKr (factor 16)": lambda lin, dev: LoKrModule("p", lin, 16, 1.0, False, 16, False, False, False, False, dev),
        "LoKr vec trick": lambda lin, dev: LoKrModule("p", lin, 16, 1.0, False, 16, False, False, False, False, dev,
                                                      lokr_vec_trick=True),
        "LoKr + DoRA": lambda lin, dev: LoKrModule("p", lin, 16, 1.0, False, 16, False, True, False, False, dev),
        "LoHa": lambda lin, dev: LoHaModule("p", lin, 16, 1.0),
        "OFT v2": lambda lin, dev: OFTModule("p", lin, 32, False, False, dropout_probability=0.0),
    }


def _build_adapter(ctx: Ctx, factory, device: torch.device, in_f: int, out_f: int, base_dtype: torch.dtype):
    torch.manual_seed(ctx.seed)  # the adapters initialize from the global generator: same start for every build
    lin = nn.Linear(in_f, out_f)
    with torch.no_grad():
        lin.weight.copy_(_rand(ctx, out_f, in_f, scale=in_f ** -0.5, seed=1))
        lin.bias.copy_(_rand(ctx, out_f, scale=0.1, seed=2))
    lin.requires_grad_(False)
    adapter = factory(lin, torch.device("cpu"))
    with torch.no_grad():  # move every adapter weight off its init (LoRA up = 0) so the delta is exercised
        for i, p in enumerate(adapter.parameters()):
            p.add_(_rand(ctx, *p.shape, scale=0.02, seed=10 + i))
    lin.to(device, base_dtype)
    adapter.to(device)
    if hasattr(adapter, "train_device"):
        adapter.train_device = device
    adapter.hook_to_module()
    return lin, adapter


@check(SECTION, "adapters forward+backward (LoRA, DoRA, LoKr, LoHa, OFT)")
def adapters(ctx: Ctx, rec: Rec):
    in_f, out_f, tokens = 256, 384, 64
    x = _rand(ctx, 2, tokens, in_f, seed=3)
    g = _rand(ctx, 2, tokens, out_f, seed=4)
    on_gpu = ctx.device.type == "cuda"
    for name, factory in _adapter_factories(ctx.device).items():
        ref_lin, ref_adapter = _build_adapter(ctx, factory, torch.device("cpu"), in_f, out_f, torch.float32)
        y_ref = ref_lin(x)
        y_ref.backward(g)

        lin, adapter = _build_adapter(ctx, factory, ctx.device, in_f, out_f,
                                      torch.bfloat16 if on_gpu else torch.float32)
        with torch.autocast(ctx.device.type, dtype=torch.bfloat16, enabled=on_gpu):
            y = lin(x.to(ctx.device))
        y.float().backward(g.to(ctx.device))
        err_y = rel_err(y, y_ref)
        err_g = max(rel_err(p.grad, r.grad) for p, r in zip(adapter.parameters(), ref_adapter.parameters(),
                                                          strict=True))
        ok = finite(y) and all(finite(p.grad) for p in adapter.parameters())
        tol = 0.03 if on_gpu else 1e-4
        rec.expect(ok and err_y < tol and err_g < tol * 2,
                   f"{name}: out rel err {err_y:.1e}, grads {err_g:.1e} (limits {tol:.0e}/{tol * 2:.0e}, "
                   f"{'bf16 autocast' if on_gpu else 'fp32'} vs fp32 CPU)")

    if ctx.full and on_gpu:
        in_f = out_f = 3072
        x = _rand(ctx, 2, 1024, in_f, seed=5).to(ctx.device, torch.bfloat16)
        base = nn.Linear(in_f, out_f).to(ctx.device, torch.bfloat16).requires_grad_(False)
        x.requires_grad_()
        t_base = ctx.bench(lambda: base(x).float().sum().backward())
        rec.metric("fwd+bwd ms 2x1024x3072 no adapter", t_base)
        for name, factory in _adapter_factories(ctx.device).items():
            lin, _ = _build_adapter(ctx, factory, ctx.device, in_f, out_f, torch.bfloat16)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                t = ctx.bench(lambda lin=lin: lin(x).float().sum().backward())
            rec.metric(f"fwd+bwd ms 2x1024x3072 {name}", t)


def _optimizer_list(ctx: Ctx):
    from modules.util.enum.Optimizer import Optimizer
    from modules.util.optimizer_util import adv_state_precisions

    optimizers = [Optimizer.ADOPT_ADV, Optimizer.PRODIGY_ADV, Optimizer.ADAMW]
    if ctx.full:
        optimizers += [Optimizer.ADAMW_ADV, Optimizer.PRODIGY_PLUS_SCHEDULE_FREE, Optimizer.SCHEDULE_FREE_ADAMW]
        if ctx.is_gpu:
            optimizers.append(Optimizer.ADAMW_8BIT)
    out = []
    for opt in optimizers:
        if opt.is_adv_optm:
            precisions = adv_state_precisions(opt) if ctx.full else ["auto", "int8_sr"]
            out += [(opt, p) for p in precisions]
        else:
            out.append((opt, None))
    return out


class _OptimizerRun:
    """one optimizer on a LoRA-like parameter set, built the way training builds it"""

    SHAPES = [(16, 512), (512, 16), (512, 1), (512,)]

    def __init__(self, ctx: Ctx, optimizer, precision, device, dtype, params=None, state_dict=None):
        from modules.util import create
        from modules.util.config.TrainConfig import TrainConfig
        from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection
        from modules.util.optimizer_util import default_optimizer_config
        from modules.util.torch_util import optimizer_to_device_

        self.ctx, self.optimizer_enum, self.device = ctx, optimizer, device
        config = TrainConfig.default_values()
        config.optimizer = default_optimizer_config(optimizer)
        if precision is not None:
            config.optimizer.state_precision = precision
        config.learning_rate = 1.0 if "PRODIGY" in str(optimizer) else 1e-3
        config.learning_rate_warmup_steps = 0
        config.train_device = str(device)
        if params is None:
            params = [_rand(ctx, *s, scale=0.1, seed=20 + i) for i, s in enumerate(self.SHAPES)]
        self.params = [nn.Parameter(p.detach().clone().to(device, dtype)) for p in params]
        self.collection = NamedParameterGroupCollection()
        self.collection.add_group(NamedParameterGroup("lora", self.params, None, "lora"))
        self.opt = create.create_optimizer(self.collection, state_dict, config)
        optimizer_to_device_(self.opt, device)
        if hasattr(self.opt, "train"):
            self.opt.train()  # schedule-free optimizers

    def step(self, i: int):
        for j, p in enumerate(self.params):
            p.grad = _rand(self.ctx, *p.shape, seed=1000 * i + j).to(self.device, p.dtype)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)

    def saved_state(self) -> dict:
        # like InternalModelSaverMixin: the extra keys, then a torch.save / torch.load round trip
        state = self.opt.state_dict()
        state["param_group_mapping"] = self.collection.unique_name_mapping
        state["param_group_optimizer_mapping"] = [str(self.optimizer_enum)]
        if self.optimizer_enum.is_adv_optm:
            import adv_optm
            state["adv_optm_version"] = getattr(adv_optm, "__version__", "unknown")
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        return torch.load(buffer, weights_only=True)

    def weights(self) -> list[torch.Tensor]:
        if hasattr(self.opt, "eval"):
            self.opt.eval()
        w = [p.detach().float().cpu().clone() for p in self.params]
        if hasattr(self.opt, "train"):
            self.opt.train()
        return w


@check(SECTION, "optimizers step, state precision and save/resume (create_optimizer)")
def optimizers(ctx: Ctx, rec: Rec):
    dtypes = [torch.float32, torch.bfloat16]
    for optimizer, precision in _optimizer_list(ctx):
        label_opt = f"{optimizer}{'' if precision is None else f' {precision}'}"
        for dtype in dtypes:
            label = f"{label_opt} {str(dtype)[6:]}"
            try:
                run = _OptimizerRun(ctx, optimizer, precision, ctx.device, dtype)
            except Exception as e:
                rec.fail(f"{label}: create_optimizer failed: {type(e).__name__}: {e}")
                continue
            for i in range(3):
                run.step(i)
            resumed = _OptimizerRun(ctx, optimizer, precision, ctx.device, dtype, params=run.params,
                                    state_dict=run.saved_state())
            for i in range(3, 6):
                run.step(i)
                resumed.step(i)
            w, w_resumed = run.weights(), resumed.weights()
            ok = all(finite(t) for t in w + w_resumed)
            diff = max(rel_err(a, b) for a, b in zip(w_resumed, w, strict=True))
            stochastic = precision is not None and precision.endswith("_sr") or dtype == torch.bfloat16
            limit = 1e-2 if stochastic else 1e-6
            if not ok:
                rec.fail(f"{label}: NaN/inf in the weights")
            elif diff > limit:
                rec.fail(f"{label}: resumed run differs from the uninterrupted one by rel {diff:.1e} (limit {limit})")
            else:
                rec.line(f"ok: {label}: 6 steps, resume diff {diff:.1e}")

        if ctx.device.type != "cpu" and precision in (None, "auto", "fp32"):
            # the same fp32 run on the GPU and on the CPU should end at (nearly) the same weights
            runs = {}
            for device in (ctx.device, torch.device("cpu")):
                try:
                    r = _OptimizerRun(ctx, optimizer, precision, device, torch.float32)
                    for i in range(6):
                        r.step(i)
                except RuntimeError as e:
                    if device.type != "cpu":
                        raise
                    # e.g. bitsandbytes' 8-bit optimizers run only on the GPU
                    rec.line(f"{label_opt}: no CPU run to compare with ({str(e).splitlines()[0][:80]})")
                    break
                runs[device.type] = r.weights()
            if "cpu" not in runs:
                continue
            diff = max(rel_err(a, b) for a, b in zip(runs["cuda"], runs["cpu"], strict=True))
            rec.expect(diff < 1e-3, f"{label_opt} fp32: GPU vs CPU after 6 steps rel diff {diff:.1e} (limit 1e-3)")


class _Block(nn.Module):
    def __init__(self, dim, expansion, adapter):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim * expansion)
        self.lin2 = nn.Linear(dim * expansion, dim)
        self._adapter = [adapter]  # kept out of the module tree, like a LoRA

    def forward(self, hidden_states, temb):
        h = self.lin2(F.gelu(self.lin1(hidden_states)))
        return hidden_states + h * (temb + self._adapter[0])


class _Toy(nn.Module):
    def __init__(self, dim, layers, expansion):
        super().__init__()
        self.adapters = nn.ParameterList([nn.Parameter(torch.randn(dim) * 0.1) for _ in range(layers)])
        self.blocks = nn.ModuleList([_Block(dim, expansion, a) for a in self.adapters])
        self.blocks.requires_grad_(False)

    def forward(self, x, temb):
        for b in self.blocks:
            x = b(x, temb)
        return x


@check(SECTION, "layer + activation offloading through LayerOffloadConductor")
def offloading(ctx: Ctx, rec: Rec):
    from modules.util.checkpointing_util import enable_checkpointing
    from modules.util.config.TrainConfig import TrainConfig, TrainModelPartConfig

    dim, layers, expansion = (1024, 12, 4) if ctx.full else (512, 8, 4)
    variants = [(f, a, s) for f in (0.5, 0.9) for a in (False, True) for s in (True, False)]
    if not ctx.full:
        variants = [(0.5, True, True), (0.9, False, True), (0.5, False, False)]
    if not ctx.is_gpu:
        variants = [v for v in variants if not v[2]]  # transfers are synchronous without a GPU
    for fraction, act, async_on in variants:
        transfers = "async" if async_on and ctx.is_gpu else "sync"
        label = f"offload {fraction}, activations {'on' if act else 'off'}, {transfers} transfers"
        torch.manual_seed(0)
        model = _Toy(dim, layers, expansion)
        ref = copy.deepcopy(model).to(ctx.device)
        originals = [copy.deepcopy(b.state_dict()) for b in model.blocks]
        model.adapters.to(ctx.device)
        config = TrainConfig.default_values()
        # on a CPU-only run, "cpu:1" is a different device to the conductor (same memory): it runs its real logic
        temp_device = "cpu:1" if ctx.device.type == "cpu" else "cpu"
        config.train_device, config.temp_device, config.async_offloading = str(ctx.device), temp_device, async_on
        part = TrainModelPartConfig.default_values()
        part.offload_fraction, part.activation_offloading, part.gradient_checkpointing = fraction, act, True
        conductor = enable_checkpointing(model, config, part, False, [(model.blocks, ["hidden_states"])])
        conductor.materialize()
        worst = 0.0
        for step in range(4):
            x = _rand(ctx, 2, [256, 1024, 64, 512][step], dim, seed=step).to(ctx.device)
            temb = _rand(ctx, dim, seed=100 + step).to(ctx.device)
            for p in list(model.adapters) + list(ref.adapters):
                p.grad = None
            loss = model(x, temb).pow(2).mean()
            loss.backward()
            ref_loss = ref(x, temb).pow(2).mean()
            ref_loss.backward()
            worst = max(worst, rel_err(loss, ref_loss),
                        *(rel_err(a.grad, b.grad) for a, b in zip(model.adapters, ref.adapters, strict=True)))
        rec.expect(worst < 1e-4, f"{label}: loss/grads vs no offloading rel err {worst:.1e}")
        t_off = ctx.bench(lambda model=model, x=x, temb=temb: model(x, temb).pow(2).mean().backward(), iters=5)
        t_ref = ctx.bench(lambda ref=ref, x=x, temb=temb: ref(x, temb).pow(2).mean().backward(), iters=5)
        rec.metric(f"{label}: step ms", t_off)
        rec.metric(f"{label}: step ms without offloading", t_ref)

        identical = True
        for _ in range(3):
            conductor.evict()
            conductor.materialize()
        conductor.evict()
        for block, original in zip(model.blocks, originals, strict=True):
            for key, value in block.state_dict().items():
                identical &= torch.equal(value.cpu(), original[key])
        rec.expect(identical, f"{label}: weights bit-identical after 3 materialize/evict cycles")
        del model, ref, conductor


@check(SECTION, "pinned host memory and transfer speed", gpu_only=True)
def pinned_memory(ctx: Ctx, rec: Rec):
    from modules.util.torch_util import pin_tensor_, unpin_tensor_

    size = 256 * 2**20
    host = torch.empty(size // 4, dtype=torch.float32)
    pin_tensor_(host)  # cudaHostRegister / hipHostRegister, as the offload caches do
    rec.expect(host.is_pinned(), "pin_tensor_ registers the memory as pinned")
    device_buf = torch.empty_like(host, device=ctx.device)
    pageable = torch.empty_like(host)
    for name, src in (("pinned", host), ("pageable", pageable)):
        ms = ctx.bench(lambda src=src: device_buf.copy_(src, non_blocking=True), iters=10)
        rec.metric(f"host->GPU GB/s ({name})", size / ms / 1e6)
        ms = ctx.bench(lambda src=src: src.copy_(device_buf, non_blocking=True), iters=10)
        rec.metric(f"GPU->host GB/s ({name})", size / ms / 1e6)
    ctx.sync()
    unpin_tensor_(host)
    rec.expect(not host.is_pinned(), "unpin_tensor_ releases it")


def _vae_from_file(path: str):
    """a VAE from one .safetensors file in the original (ComfyUI, A1111, BFL ae.safetensors) or diffusers layout, also
    inside a full checkpoint. The config is read from the tensor shapes: diffusers' from_single_file fetches it from
    Hugging Face instead, which fails offline and for gated repos."""
    from diffusers import AutoencoderKL
    from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint

    from safetensors import safe_open

    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        prefix = next((p for p in ("", "first_stage_model.", "vae.") if p + "encoder.conv_in.weight" in keys), None)
        if prefix is None:
            raise ValueError(f"{path} holds no VAE weights (no encoder.conv_in.weight): it is probably a transformer or "
                             "UNet only file. Pass the VAE file (e.g. ComfyUI models/vae/ae.safetensors for "
                             "Flux/Chroma) or a diffusers model folder with vae/")
        parts = tuple(prefix + part for part in ("encoder.", "decoder.", "quant_conv.", "post_quant_conv."))
        state = {k[len(prefix):]: f.get_tensor(k) for k in keys if k.startswith(parts)}

    diffusers_layout = "encoder.down_blocks.0.resnets.0.conv1.weight" in state
    block = "encoder.down_blocks.{}.resnets.0.conv2.weight" if diffusers_layout else "encoder.down.{}.block.0.conv2.weight"
    block_out_channels = []
    while block.format(len(block_out_channels)) in state:
        block_out_channels.append(state[block.format(len(block_out_channels))].shape[0])
    resnet = "encoder.down_blocks.0.resnets.{}.conv1.weight" if diffusers_layout else "encoder.down.0.block.{}.conv1.weight"
    layers_per_block = 0
    while resnet.format(layers_per_block) in state:
        layers_per_block += 1
    config = {
        "in_channels": state["encoder.conv_in.weight"].shape[1],
        "out_channels": state["decoder.conv_out.weight"].shape[0],
        "latent_channels": state["encoder.conv_out.weight"].shape[0] // 2,
        "down_block_types": ["DownEncoderBlock2D"] * len(block_out_channels),
        "up_block_types": ["UpDecoderBlock2D"] * len(block_out_channels),
        "block_out_channels": block_out_channels,
        "layers_per_block": layers_per_block,
        "use_quant_conv": "quant_conv.weight" in state,
        "use_post_quant_conv": "post_quant_conv.weight" in state,
    }
    if not diffusers_layout:
        state = convert_ldm_vae_checkpoint(state, config)
    vae = AutoencoderKL(**config)
    vae.load_state_dict(state, strict=True)
    return vae.float()


def _vae(ctx: Ctx, device: torch.device | None = None):
    """the VAE for the VAE checks, in fp32 as training caches with it: --vae if given, else the 16-channel layout of
    the Flux/Chroma/SD3-era models, randomly initialized (no download needed)"""
    from diffusers import AutoencoderKL

    device = device or ctx.device
    if ctx.vae_path:
        if os.path.isfile(ctx.vae_path):
            vae = _vae_from_file(ctx.vae_path)
        elif os.path.isdir(os.path.join(ctx.vae_path, "vae")):
            vae = AutoencoderKL.from_pretrained(ctx.vae_path, subfolder="vae", torch_dtype=torch.float32)
        else:
            vae = AutoencoderKL.from_pretrained(ctx.vae_path, torch_dtype=torch.float32)
    else:
        torch.manual_seed(0)
        vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=16,
                            down_block_types=["DownEncoderBlock2D"] * 4, up_block_types=["UpDecoderBlock2D"] * 4,
                            block_out_channels=[128, 256, 512, 512], layers_per_block=2,
                            use_quant_conv=False, use_post_quant_conv=False)
    return vae.to(device).eval().requires_grad_(False)


def _vae_image(ctx: Ctx, res: int, seed: int) -> torch.Tensor:
    # a smooth random image in [-1, 1]: closer to a photo than per-pixel noise
    low = torch.rand(1, 3, max(res // 16, 2), max(res // 16, 2), generator=ctx.generator(seed))
    return (F.interpolate(low, size=(res, res), mode="bicubic", align_corners=False).clamp(0, 1) * 2 - 1)


_LAYER_TYPES = (nn.Conv2d, nn.GroupNorm, nn.Linear)


def _encode_with_layer_errors(vae, img: torch.Tensor, reference: dict | None):
    """encodes img; with a reference (layer name -> CPU output), returns each layer's rel error to it in call order.
    Without one, returns the layer outputs themselves (as the reference)."""
    from diffusers.models.attention_processor import Attention

    records, hooks = [], []
    for name, module in vae.encoder.named_modules():
        if isinstance(module, _LAYER_TYPES + (Attention,)):
            def hook(mod, args, out, name=name):
                out = out[0] if isinstance(out, tuple) else out
                if reference is None:
                    records.append((name, out.detach().float().cpu().clone()))
                else:
                    records.append((name, type(mod).__name__, rel_err(out, reference[name]), finite(out)))
            hooks.append(module.register_forward_hook(hook))
    try:
        with torch.no_grad():
            latent = vae.encode(img).latent_dist.mean.float().cpu()
    finally:
        for h in hooks:
            h.remove()
    return (dict(records) if reference is None else records), latent


@check(SECTION, "VAE encode per layer: main thread vs worker threads, MIOpen and attention toggles",
       note="compares every layer of the VAE encoder against a CPU reference; the CPU part takes a while")
def vae_layers(ctx: Ctx, rec: Rec):
    """caching runs the VAE in worker threads. This finds the first layer (if any) whose output differs from a CPU
    float64-checked reference, per run, and which runtime toggle makes the difference go away."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    res = 128 if ctx.device.type == "cpu" else (512 if ctx.full else 256)
    img = _vae_image(ctx, res, seed=7)
    rec.line(f"{'VAE ' + ctx.vae_path if ctx.vae_path else 'random-init 16-channel VAE'}, fp32, {res}x{res} image")

    reference, ref_latent = _encode_with_layer_errors(_vae(ctx, torch.device("cpu")), img, None)
    vae = _vae(ctx)
    img_d = img.to(ctx.device)
    miopen_default = torch.backends.cudnn.enabled

    def run(label: str, thread: bool, miopen: bool = True, math_attention: bool = False, sdpa_fix: bool = True):
        out = {}

        def body():
            try:
                if math_attention:
                    with sdpa_kernel([SDPBackend.MATH]):
                        out["r"] = _encode_with_layer_errors(vae, img_d, reference)
                else:
                    out["r"] = _encode_with_layer_errors(vae, img_d, reference)
            except Exception as e:
                out["e"] = f"{type(e).__name__}: {e}"

        torch.backends.cudnn.enabled = miopen
        fix_was_installed = rocm_sdpa_fix.installed()
        if not sdpa_fix:
            rocm_sdpa_fix.uninstall()
        try:
            if thread:
                t = threading.Thread(target=body)
                t.start()
                t.join()
            else:
                body()
        finally:
            torch.backends.cudnn.enabled = miopen_default
            if fix_was_installed:
                rocm_sdpa_fix.install()
        if "e" in out:
            rec.fail(f"{label}: {out['e']}")
            return None
        records, latent = out["r"]
        final = rel_err(latent, ref_latent)
        bad = [r for r in records if not r[3] or r[2] > 1e-3]
        if not bad and final < 1e-3:
            rec.line(f"ok: {label}: every layer matches the CPU (latent rel err {final:.1e})")
            return True
        name, kind, err, is_finite = bad[0] if bad else ("(none)", "", final, True)
        types = sorted({r[1] for r in bad})
        report = rec.warn if not sdpa_fix and rocm_sdpa_fix.applies_to(ctx.device) else rec.fail
        report(f"{label}: latent rel err {final:.2e}; first wrong layer {name} ({kind}): "
                 f"{'NaN/inf' if not is_finite else f'rel err {err:.2e}'}; wrong layer types: {', '.join(types)}")
        return False

    results = {
        "main": run("main thread", False),
        "main again": run("main thread, again", False),
        "worker": run("worker thread", True),
        "worker again": run("another worker thread", True),
    }
    if ctx.is_gpu:
        results["worker no MIOpen"] = run("worker thread, MIOpen off (torch.backends.cudnn.enabled=False)", True,
                                          miopen=False)
        results["worker math attention"] = run("worker thread, SDPA math kernel", True, math_attention=True)
        results["main no MIOpen"] = run("main thread, MIOpen off", False, miopen=False)
        if rocm_sdpa_fix.applies_to(ctx.device):
            run("main thread, without OneTrainer's attention fix (rocm_sdpa_fix; expected wrong here)", False,
                sdpa_fix=False)
    fixes = [k for k, v in results.items() if v and not results.get(k.split(" no ")[0].split(" math")[0], True)]
    if fixes:
        rec.line("the runs that match where the plain run doesn't: " + ", ".join(fixes))


@check(SECTION, "concurrent VAE encodes (caching with dataloader_threads > 1)")
def concurrent_encodes(ctx: Ctx, rec: Rec):
    """caching encodes images in several threads on the same GPU; on ROCm that once gave random all-NaN latents.
    OneTrainer now encodes one at a time (DataLoaderMgdsMixin._run_models_one_at_a_time). This measures whether
    unlocked concurrent encodes still go wrong on this GPU, and that locked ones don't."""
    encoder = _vae(ctx)
    res = 512 if ctx.full else 256
    if ctx.device.type == "cpu":
        res = 128
    count = 8 if ctx.full else 4
    rounds = 6 if ctx.full else 2
    images = [_vae_image(ctx, res, seed=i).to(ctx.device) for i in range(count)]

    def encode(img):
        with torch.no_grad():
            return encoder.encode(img).latent_dist.mean

    reference = [encode(img).cpu() for img in images]
    serial_bad = []
    for i, (img, ref) in enumerate(zip(images, reference, strict=True)):
        again = encode(img)
        if not finite(ref) or not finite(again):
            serial_bad.append(f"image {i}: NaN/inf")
        elif rel_err(again, ref) > 1e-3:
            serial_bad.append(f"image {i}: rel err {rel_err(again, ref):.1e}")
    rec.expect(not serial_bad, f"serial encodes finite and repeatable in the main thread ({len(serial_bad)} bad of "
                               f"{count}{': ' + ', '.join(serial_bad) if serial_bad else ''})")

    for threads in (2, 4):
        for locked in (False, True):
            lock = threading.Lock()
            bad, errors = [], []

            def worker(t, lock=lock, bad=bad, errors=errors, locked=locked, threads=threads):
                try:
                    for r in range(rounds):
                        for i in range(t, count, threads) if r % 2 else range(count):
                            if locked:
                                with lock:
                                    latent = encode(images[i])
                            else:
                                latent = encode(images[i])
                            if not finite(latent):
                                bad.append((i, "NaN/inf"))
                            elif rel_err(latent, reference[i]) > 1e-3:
                                bad.append((i, f"rel err {rel_err(latent, reference[i]):.1e}"))
                except Exception as e:
                    errors.append(f"{type(e).__name__}: {e}")

            t0 = time.perf_counter()
            pool = [threading.Thread(target=worker, args=(t,)) for t in range(threads)]
            for t in pool:
                t.start()
            for t in pool:
                t.join()
            seconds = time.perf_counter() - t0
            label = f"{threads} threads, {'locked' if locked else 'unlocked'}"
            rec.metric(f"{label}: seconds", seconds)
            if errors:
                rec.fail(f"{label}: {errors[0]}")
            elif bad:
                msg = f"{label}: {len(bad)} bad latents, e.g. image {bad[0][0]}: {bad[0][1]}"
                if locked:
                    rec.fail(msg)
                else:
                    rec.warn(msg + " -- concurrent encodes break on this GPU; keep the caching lock")
            else:
                rec.line(f"ok: {label}: all latents finite and matching")
