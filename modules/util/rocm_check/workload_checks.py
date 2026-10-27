"""The main workload end to end, at a tiny scale: a Chroma model with random weights and a small masked dataset, built
offline (tiny_chroma.py), trained through the real scripts/train.py with the workload's settings (Chroma LoRA preset,
masked training, EMA, ADOPT_ADV, no warmup, sampling, backups). Each variant that changes how GPU memory is handled
(MemPools, disk streaming, layer offloading, resuming a backup) is compared with the baseline and with a repeat of it,
which measures the run-to-run noise; every run writes an OT_DIAG_LOG, whose memory numbers are reported. And the same
model's components (T5, VAE, transformer with a LoRA) on the GPU against the CPU, with the same inputs.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

from modules.util.rocm_check.framework import Ctx, Rec, check, finite, rel_err

import torch

SECTION = "5 training run (tiny Chroma)"
_SEED = "42"
_PRESET = os.path.join("training_presets", "Chroma", "#chroma LoRA 24GB.json")


def _repo_root(ctx: Ctx) -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(ctx.script_path)))


def _base_config(ctx: Ctx, model: str, data: str, run_dir: str, epochs: int) -> dict:
    with open(os.path.join(_repo_root(ctx), _PRESET)) as f:
        config = json.load(f)
    gpu = ctx.device.type == "cuda"
    with open(os.path.join(run_dir, "concepts.json"), "w") as f:
        json.dump([{"name": "data", "path": data, "enabled": True, "seed": 1, "text": {"prompt_source": "sample"}}], f)
    with open(os.path.join(run_dir, "samples.json"), "w") as f:
        json.dump([{"prompt": "a red cat on a table", "height": 256, "width": 256, "seed": 7, "diffusion_steps": 4,
                    "cfg_scale": 1.0, "enabled": True}], f)
    config.update({
        "__version": 13, "base_model_name": model, "resolution": "256", "epochs": epochs,
        # the main workload: masked training, EMA, ADOPT_ADV, no warmup (the preset gives the rest: LoRA, bf16, batch 2)
        "masked_training": True, "ema": "GPU" if gpu else "CPU", "ema_update_step_interval": 1,
        "optimizer": {"optimizer": "ADOPT_ADV"}, "learning_rate_warmup_steps": 0,
        "train_device": str(ctx.device), "temp_device": "cpu",
        "sample_after": 1, "sample_after_unit": "EPOCH", "sample_image_format": "PNG",
        "backup_after": 1, "backup_after_unit": "EPOCH", "tensorboard": False, "tensorboard_always_on": False,
        "workspace_dir": os.path.join(run_dir, "workspace"), "cache_dir": os.path.join(run_dir, "cache"),
        "final_output_dir": os.path.join(run_dir, "output"), "run_name_mode": "CUSTOM", "run_name": "lora",
        "concept_file_name": os.path.join(run_dir, "concepts.json"),
        "sample_definition_file_name": os.path.join(run_dir, "samples.json"),
    })
    return config


def _train(ctx: Ctx, run_dir: str, config: dict, env: dict[str, str], values: list[str], log_name: str) -> dict:
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=1)
    diag = os.path.join(run_dir, f"{log_name}.jsonl")
    cmd = [sys.executable, os.path.join(_repo_root(ctx), "scripts", "train.py"), "--config-path", config_path]
    for value in values:
        cmd += ["--config-value", value]
    run_env = {**os.environ, "OT_SEED": _SEED, **env, "OT_DIAG_LOG": diag, "OT_DIAG_LABEL": log_name}
    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=_repo_root(ctx), env=run_env, capture_output=True, text=True, timeout=1800)
    with open(os.path.join(run_dir, f"{log_name}.log"), "w") as f:
        f.write(result.stdout + result.stderr)
    tail = [line for line in (result.stdout + result.stderr).splitlines() if line.strip()][-6:]
    return {"rc": result.returncode, "seconds": time.perf_counter() - start, "diag": diag, "tail": tail,
            "output": os.path.join(run_dir, "output", "lora.safetensors"),
            "samples": os.path.join(run_dir, "workspace", "samples")}


def _weights(path: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    return {k: v.double() for k, v in load_file(path).items()}


def _weights_diff(a: dict, b: dict) -> float:
    if a.keys() != b.keys():
        return float("inf")
    return max(((a[k] - b[k]).norm() / a[k].norm().clamp_min(1e-30)).item() for k in a)


def _losses(diag: str) -> dict[int, float]:
    out = {}
    if os.path.exists(diag):
        with open(diag) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    break
                if r["kind"] == "header":
                    out = {}  # a later run appended to the log: its losses only
                elif r["kind"] == "step":
                    out[r["step"]] = r["loss"]
    return out


def _loss_diff(a: dict, b: dict) -> tuple[float, int | None, int]:
    common = sorted(set(a) & set(b))
    worst = max((abs(a[s] - b[s]) / max(abs(a[s]), 1e-12) for s in common), default=0.0)
    first = next((s for s in common if a[s] != b[s]), None)
    return worst, first, len(common)


def _sample_images(path: str) -> tuple[int, int]:
    import numpy as np
    from PIL import Image

    images = [os.path.join(r, n) for r, _, ns in os.walk(path) for n in ns if n.endswith(".png")] \
        if os.path.isdir(path) else []
    blank = sum(1 for p in images if np.asarray(Image.open(p)).astype(np.float32).std() < 1.0)
    return len(images), blank


def _variants(ctx: Ctx) -> list[tuple[str, dict, list[str]]]:
    offload = ["transformer.offload_fraction=0.5", "transformer.activation_offloading=true"]
    if not ctx.is_gpu:
        return [("repeat", {}, [])]
    variants = [("repeat", {}, [])]
    if ctx.is_rocm:  # MemPools are always used on NVIDIA
        variants += [("MemPools", {"OT_ROCM_MEM_POOLS": "true"}, []),
                     ("MemPools + offloading", {"OT_ROCM_MEM_POOLS": "true"}, offload)]
    variants += [("streaming off", {}, ["stream_from_disk=false"]),
                 ("layer offloading 0.5 + activations", {}, offload)]
    if ctx.full:
        variants += [("cache in RAM off", {}, ["transformer.cache_in_ram=false", "text_encoder.cache_in_ram=false"]),
                     ("simplex offloading", {}, ["transformer.offload_fraction=0.5",
                                                 "transformer.simplex_offloading=true"]),
                     ("synchronous offloading", {}, [*offload, "async_offloading=false"]),
                     ("deterministic algorithms", {"OT_DETERMINISTIC": "true"}, [])]
    return variants


@check(SECTION, "training runs: tiny Chroma LoRA as the main workload (repeat, MemPools, streaming, offloading, "
                "resume)", timeout=5400.0,
       note="trains a tiny random Chroma through scripts/train.py once per variant: a few minutes")
def training_runs(ctx: Ctx, rec: Rec):
    from modules.util.diag_report import load, summarize
    from modules.util.rocm_check.tiny_chroma import build_dataset, build_tiny_chroma

    rec.info()
    base_dir = tempfile.mkdtemp(prefix="rocm_check_training_", dir=ctx.out_dir if os.path.isdir(ctx.out_dir) else None)
    failed = False
    try:
        model = build_tiny_chroma(os.path.join(base_dir, "model"))
        data = build_dataset(os.path.join(base_dir, "data"), count=8, size=256)
        epochs = 3

        def run(name: str, env: dict, values: list[str], epochs: int = epochs, run_dir: str | None = None,
                log_name: str = "diag") -> dict:
            run_dir = run_dir or os.path.join(base_dir, name.replace(" ", "_").replace("+", "and"))
            os.makedirs(run_dir, exist_ok=True)
            config = _base_config(ctx, model, data, run_dir, epochs)
            config["lora_model_name"] = initial_lora
            return _train(ctx, run_dir, config, env, values, log_name)

        # every run starts from the same LoRA: a LoRA layer is initialized on the device its base layer is on, so with
        # layer offloading the offloaded layers draw from the CPU's generator instead of the GPU's, and the runs would
        # start apart even with OT_SEED. A run at learning rate 0 saves the start (one epoch, no samples or backups).
        initial_lora = ""
        init_dir = os.path.join(base_dir, "initial_lora")
        os.makedirs(init_dir, exist_ok=True)
        init = _train(ctx, init_dir, _base_config(ctx, model, data, init_dir, 1), {},
                      ["learning_rate=0", "sample_after_unit=NEVER", "backup_after_unit=NEVER",
                       "backup_before_save=false"], "diag")
        if init["rc"] != 0 or not os.path.exists(init["output"]):
            failed = True
            rec.fail(f"the run that makes the initial LoRA failed (exit code {init['rc']}): "
                     + " | ".join(init["tail"])[-600:])
            return
        initial_lora = init["output"]

        results = {"baseline": run("baseline", {}, [])}
        for name, env, values in _variants(ctx):
            results[name] = run(name, env, values)
        # resume: stop after 2 of the 3 epochs (the backup at the end), then continue from that backup
        resume_dir = os.path.join(base_dir, "resume")
        first = run("resume", {}, [], epochs=epochs - 1, run_dir=resume_dir, log_name="first")
        results["resumed from a backup"] = \
            run("resume", {}, ["continue_last_backup=true"], run_dir=resume_dir, log_name="diag") \
            if first["rc"] == 0 else first

        # every run: finished, finite weights, sample images
        weights = {}
        for name, r in results.items():
            if r["rc"] != 0 or not os.path.exists(r["output"]):
                failed = True
                rec.fail(f"{name}: training failed (exit code {r['rc']}): " + " | ".join(r["tail"])[-600:])
                continue
            w = _weights(r["output"])
            weights[name] = w
            ok = all(finite(v) for v in w.values())
            count, blank = _sample_images(r["samples"])
            rec.expect(ok and count > 0 and blank == 0,
                       f"{name}: finished in {r['seconds']:.0f} s, {len(w)} LoRA tensors "
                       f"{'finite' if ok else 'with NaN/inf'}, {count} sample images ({blank} blank)")
            failed |= not ok

        if "baseline" not in weights:
            return
        base_losses = _losses(results["baseline"]["diag"])
        floor = _weights_diff(weights["baseline"], weights["repeat"]) if "repeat" in weights else 0.0
        floor_loss = _loss_diff(base_losses, _losses(results["repeat"]["diag"]))[0] if "repeat" in weights else 0.0
        rec.line(f"run-to-run noise (baseline vs repeat): weights {floor:.1e}, losses {floor_loss:.1e}"
                 + (" (bit-identical)" if floor == 0 else ""))
        limit = max(10 * floor, 1e-6)
        for name, w in weights.items():
            if name in ("baseline", "repeat"):
                continue
            diff = _weights_diff(weights["baseline"], w)
            worst, first_step, steps = _loss_diff(base_losses, _losses(results[name]["diag"]))
            text = (f"{name} vs baseline: weights {diff:.1e}, losses {worst:.1e} over {steps} steps"
                    + (", bit-identical weights" if diff == 0 else
                       f", losses part at step {first_step}" if first_step is not None else ""))
            if diff <= limit:
                rec.line(f"ok: {text}")
            elif diff <= 1e-2:
                rec.warn(f"{text}: beyond 10x the run-to-run noise (limit {limit:.1e})")
            else:
                failed = True
                rec.fail(f"{text}: far beyond the run-to-run noise (limit {limit:.1e})")

        # the memory each variant used (tiny model: the allocator's behavior, not a 24 GB workload's numbers)
        for name, r in results.items():
            if not os.path.exists(r["diag"]):
                continue
            s = summarize(load(r["diag"]))
            if "peak reserved GiB" not in s:
                continue
            rec.line(f"{name}: peak reserved {s['peak reserved GiB'] * 1024:.0f} MiB, held unused at the end "
                     f"{(s['held unused at end GiB'] or 0) * 1024:.0f} MiB, alloc retries {s['alloc retries']}, "
                     f"{s['part moves']} part moves in {s['part move s']:.1f} s, step {s['step s median'] or 0:.3f} s")
            rec.metric(f"{name}: peak reserved MiB", s["peak reserved GiB"] * 1024)
    finally:
        if failed:
            rec.line(f"the runs are kept in {base_dir} (each run's .log and .jsonl, compare with diag_report)")
        else:
            shutil.rmtree(base_dir, ignore_errors=True)


@check(SECTION, "tiny Chroma components (T5, VAE, transformer + LoRA) on the GPU vs the CPU", gpu_only=True)
def components(ctx: Ctx, rec: Rec):
    """the model loaded through the real ChromaModelLoader on both devices, the same inputs: the text encoder through
    ChromaModel.encode_text (padding mask, cut to the longest caption), the VAE both ways, and the transformer called as
    BaseChromaSetup.predict calls it, with a LoRA on the workload's layer filter, forward and backward. fp32 on the GPU
    must match the CPU closely; bf16 weights under bf16 autocast (the workload) within bf16 precision"""
    from random import Random

    from modules.model.ChromaModel import ChromaModel
    from modules.modelLoader.chroma.ChromaModelLoader import ChromaModelLoader
    from modules.module.LoRAModule import LoRAModuleWrapper
    from modules.util.config.TrainConfig import TrainConfig
    from modules.util.enum.DataType import DataType
    from modules.util.enum.ModelType import ModelType
    from modules.util.ModelNames import ModelNames
    from modules.util.rocm_check.tiny_chroma import build_tiny_chroma

    with tempfile.TemporaryDirectory(dir=ctx.out_dir if os.path.isdir(ctx.out_dir) else None) as tmp:
        path = build_tiny_chroma(os.path.join(tmp, "model"))
        with open(os.path.join(_repo_root(ctx), _PRESET)) as f:
            preset = json.load(f)

        def load(dtype: DataType) -> tuple[ChromaModel, TrainConfig]:
            config = TrainConfig.default_values()
            config.model_type = ModelType.CHROMA_1
            config.transformer.weight_dtype = config.text_encoder.weight_dtype = dtype
            config.vae.weight_dtype = DataType.FLOAT_32
            config.train_dtype = dtype
            config.layer_filter = preset["layer_filter"]
            model = ChromaModel(ModelType.CHROMA_1)
            ChromaModelLoader().load(model, ModelType.CHROMA_1, ModelNames(base_model=path), config.weight_dtypes(),
                                     config.quantization, stream_from_disk=False)
            return model, config

        prompts = ["a red cat on a table", "photo of a small house near a river under a dark night sky with a moon"]
        g = torch.Generator().manual_seed(ctx.seed)
        image = torch.rand(2, 3, 256, 256, generator=g) * 2 - 1
        latents = torch.randn(2, 16, 32, 32, generator=g)
        timestep = torch.tensor([300.0, 800.0])
        target = torch.randn(2, 32 * 32 // 4, 64, generator=g)

        def run(model: ChromaModel, config: TrainConfig, device: torch.device, autocast: bool, lora_state=None):
            model.text_encoder.to(device)
            model.vae.to(device)
            model.transformer.to(device)
            lora = LoRAModuleWrapper(model.transformer, "transformer", config, config.layer_filter.split(","))
            if lora_state is None:
                gl = torch.Generator().manual_seed(ctx.seed + 1)
                with torch.no_grad():
                    for p in lora.parameters():
                        p.copy_(torch.randn(p.shape, generator=gl) * 0.05)
            else:
                lora.load_state_dict(lora_state)
            lora.to(device)
            lora.hook_to_module()
            dtype = config.train_dtype.torch_dtype()
            ac = torch.autocast(device.type, dtype=dtype, enabled=autocast)
            out = {}
            with torch.no_grad(), ac:
                text, mask = model.encode_text(device, batch_size=2, rand=Random(0), text=prompts)
                out["text"] = text.float().cpu()
                out["mask"] = mask.cpu()
                out["vae latent"] = model.vae.encode(image.to(device)).latent_dist.mean.float().cpu()
                out["vae image"] = model.vae.decode(latents.to(device)).sample.float().cpu()
            # the transformer gets the CPU's text embedding, so its comparison doesn't carry the text encoder's error
            text_in = out_ref["text"].to(device) if out_ref else text
            mask_in = out_ref["mask"].to(device) if out_ref else mask
            packed = model.pack_latents(latents.to(device))
            image_ids = model.prepare_latent_image_ids(32, 32, device, dtype)
            text_ids = torch.zeros(text_in.shape[1], 3, device=device)
            attention_mask = torch.cat([mask_in, torch.ones(2, packed.shape[1], dtype=torch.bool, device=device)], 1)
            with ac:
                pred = model.transformer(
                    hidden_states=packed.to(dtype), timestep=timestep.to(device) / 1000,
                    encoder_hidden_states=text_in.to(dtype), txt_ids=text_ids.to(dtype), img_ids=image_ids.to(dtype),
                    attention_mask=attention_mask, joint_attention_kwargs=None, return_dict=True).sample
                loss = torch.nn.functional.mse_loss(pred.float(), target.to(device))
            loss.backward()
            out["transformer out"] = pred.float().detach().cpu()
            out["LoRA grads"] = torch.cat([p.grad.float().flatten().cpu() for p in lora.parameters()])
            out["lora_state"] = {k: v.detach().cpu() for k, v in lora.state_dict().items()}
            return out

        out_ref = None
        cpu_model, cpu_config = load(DataType.FLOAT_32)
        out_ref = run(cpu_model, cpu_config, torch.device("cpu"), autocast=False)
        rec.expect(bool(out_ref["mask"][0].sum() < out_ref["mask"][1].sum()),
                   f"the two captions give different text lengths ({int(out_ref['mask'][0].sum())} and "
                   f"{int(out_ref['mask'][1].sum())} tokens of {out_ref['mask'].shape[1]}), so the attention is masked")
        for label, dtype, autocast, limit in (("fp32", DataType.FLOAT_32, False, 1e-4),
                                              ("bf16 weights + autocast", DataType.BFLOAT_16, True, 5e-2)):
            model, config = load(dtype)
            out = run(model, config, ctx.device, autocast, lora_state=out_ref["lora_state"])
            for key in ("text", "vae latent", "vae image", "transformer out", "LoRA grads"):
                err = rel_err(out[key], out_ref[key])
                rec.expect(finite(out[key]) and err < limit * (2 if key == "LoRA grads" else 1),
                           f"{label}: {key} rel err {err:.1e} (limit {limit * (2 if key == 'LoRA grads' else 1):.0e})")
            del model
