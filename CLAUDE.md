# CLAUDE.md — OneTrainer

## Overview
- OneTrainer: diffusion-model trainer (full fine-tune, LoRA/LoHa/LoKr/OFTv2, embeddings) for SD1.5 through Flux/Chroma/Qwen/etc. Has a Qt GUI and a headless CLI.
- Built on diffusers + a custom graph data loader, **mgds** (pinned git commit in `requirements-global.txt`).
- User docs: [README.md](README.md), [LAUNCH-SCRIPTS.md](LAUNCH-SCRIPTS.md), [docs/](docs/). Don't repeat them here.

## Commands
Run from the repo root. Launchers `cd` there themselves, but raw `python scripts/...` must be started from the root: `factory.import_dir("modules/...")`, `resources/`, `secrets.json` and the default workspace paths are all relative to CWD.
```sh
./install.sh                     # create venv (default) or conda env, install requirements-global + platform file
OT_PLATFORM_REQUIREMENTS=requirements-rocm.txt ./install.sh   # skip GPU auto-detect (it picks /dev/nvidia0 before /dev/kfd)
./update.sh                      # git pull + pip install --upgrade --upgrade-strategy eager (re-pins torch!)
./start-ui.sh                    # PySide6 UI -> scripts/train_ui_qt.py
./run-cmd.sh train_ui_ctk        # legacy customtkinter UI (still maintained, same Base* views)
./run-cmd.sh train --config-path cfg.json                         # headless training
./run-cmd.sh train --preset-path "training_presets/Chroma/#chroma LoRA 24GB.json" \
    --config-path cfg.json --config-value epochs=1 --config-value transformer.train=true
./run-cmd.sh create_train_files --config-output-destination c.json \
    --concepts-output-destination concepts.json --samples-output-destination samples.json
./run-cmd.sh <script> -h         # any file in scripts/ (sample, convert_model, generate_masks, ...)
./run-cmd.sh generate_debug_report
ruff check .                     # lint config in pyproject.toml; E501 ignored, line-length 120
pre-commit run --all-files       # pre-commit hooks + ruff --fix (linter only, NOT ruff format)
```
- **No test suite.** No GitHub Actions workflows; only pre-commit.ci. Validate changes with `ruff check` plus a short real run (e.g. `--config-value epochs=1`).
- `--config-value KEY=VALUE`: dotted keys walk nested `BaseConfig`s (`optimizer.beta1=0.9`). The help text's example `ema.decay` is **wrong**: `ema` is an `EMAMode` enum; the field is `ema_decay`.
- `--preset-path` turns off migration for **both** files, so the config must already be in the current format (`__version` 11).
- Python: min 3.10, <3.14; conda path uses 3.13 (`lib.include.sh:35-37`). In the script, `OT_PREFER_VENV` defaults to `true`, although LAUNCH-SCRIPTS.md says `false`.

## Architecture (one training run)
```
scripts/train.py:main                      (CLI)          scripts/train_ui_qt.py -> modules/ui/PySide6TrainUIView.py (UI)
  TrainConfig.default_values()                             TrainUIController.start_training:282 writes training_presets/#.json
  .from_dict(preset, migrate=False)                          + secrets.json, trains in a thread on the *live* config object
  .from_dict(config, migrate=preset is None)
  --config-value overrides; secrets.json -> config.secrets
  -> modules/util/create.py:create_trainer:1372  (cloud.enabled -> CloudTrainer | multi_gpu -> MultiTrainer | GenericTrainer)

modules/trainer/GenericTrainer.py
  start():88
    create_model_loader  -> factory.get(BaseModelLoader, model_type, training_method)
                            e.g. modelLoader/ChromaLoRAModelLoader.py (make_lora_model_loader) -> modelLoader/chroma/*
    model_setup.setup_optimizations  -> per-part quantize / checkpointing / offload, attention backend (BaseModelSetup._setup_model_part)
    model_setup.setup_train_device
    model_setup.setup_model          -> LoRAModuleWrapper(...) per part, load lora_state_dict, hook_to_module()
                                        -> optimizer_util.init_model_parameters -> create.create_optimizer:128 + create_ema:1086
    create_data_loader   -> dataLoader/<Model>BaseDataLoader.py (registered per ModelType only)
                            DataLoaderText2ImageMixin._create_dataset:375 -> DataLoaderMgdsMixin._create_mgds -> mgds.MGDS/TrainDataLoader
    create_model_saver / create_model_sampler
  train():616   per epoch: dataset.start_next_epoch() (does caching) -> lr scheduler created lazily (create_lr_scheduler:1111)
                per batch: setup.predict -> setup.calculate_loss -> backward -> (update step) clip_grad_norm,
                           optimizer.step (or fused back pass hooks) -> lr_scheduler.step -> after_optimizer_step -> ema.step
                sample/backup/save go through TrainCommands; NaN loss raises RuntimeError
  end():847     backup (if backup_before_save) -> copy EMA into params -> model_saver.save(output_model_destination)
```
- mgds stage order (`_create_dataset`): enumerate → load → mask aug → aspect bucketing → crop → augment → inpainting → **preparation** (model: VAE encode, tokenize, TE encode) → **cache** (`DiskCache` under `cache_dir/image`, `cache_dir/text`) → output (aspect batch sorting). The model-specific parts are `_preparation_modules`, `_cache_modules` and `_output_modules` in each `<Model>BaseDataLoader`. mgds source: `python -c "import mgds; print(mgds.__file__)"`.
- Per-model components (unet/transformer/text_encoder_N/vae…): `ModelType._MODEL_PARTS` (`modules/util/enum/ModelType.py:286`) is the single source of truth.
- Loss: `modelSetup/mixin/ModelSetupDiffusionLossMixin.py` (masked vs unmasked, flow-matching weighting); masked math is in `modules/util/loss/masked_loss.py`.
- Adapters: `modules/module/LoRAModule.py` (`LoRAModule`, `DoRAModule`, `LoHaModule`, `LoKrModule`, `OFTModule`, `LoRAModuleWrapper:833`). Export key conversion is in `modules/util/convert_lora_util.py`, and per-model savers are in `modelSaver/<model>/<Model>LoRASaver.py`.

## Extension checklists
### Registration mechanics (read first)
- `modules/util/factory.py`: `@factory.register(Base, ModelType.X[, TrainingMethod.Y])`, or a `make_*_model_{loader,saver}()` helper that registers internally.
- `create.py:53-57` auto-imports **only top-level files** in `modelSampler/ modelLoader/ modelSaver/ modelSetup/ dataLoader/`. The repo has no `__init__.py` files, so subfolders such as `modelLoader/chroma/` are never scanned. Registration must live in a top-level file.
- `factory.get` returns `None` when nothing is registered. The failure then shows up later as a `NoneType` error. DataLoader and Sampler lookups fall back to `(model_type)` without a training method.

### New optimizer
1. `modules/util/enum/Optimizer.py`: add the member. Update `is_adaptive`, `is_schedule_free` and `supports_fused_back_pass()` if they apply.
2. `modules/util/create.py:create_optimizer`: add a `case`. The `match` has **no default**, so an unhandled enum gives `optimizer=None`.
3. `modules/util/optimizer_util.py:OPTIMIZER_DEFAULT_PARAMETERS[Optimizer.X]` is **required**; `change_optimizer` raises KeyError when the optimizer is picked in the UI. Its keys decide which params the UI shows.
4. A new hyperparameter needs: a `TrainOptimizerConfig` annotation plus a `default_values()` entry (`TrainConfig.py:35`/`:146`), a `KEY_DETAIL_MAP` entry in `modules/ui/BaseOptimizerParamsWindowView.py:32` (keys missing there are silently **hidden**), and use in `create.py`.
5. Pin the package in `requirements-global.txt`. If it is CUDA-only (bnb etc.), say so; `requirements-rocm.txt` has no bitsandbytes.

### New adapter (PEFT) type
1. `PeftType` enum in `modules/util/enum/ModelType.py` (bottom of file).
2. `LoRAModule.py`: a `PeftBase` subclass, `DummyXModule = XModule.make_dummy()` (~line 826), and a branch in `LoRAModuleWrapper.__init__` (~line 874) that sets `klass/dummy_klass/additional_args/additional_kwargs`.
3. Config fields in `TrainConfig` (see "New config field").
4. UI: `LoraTabController.get_peft_types()` and a branch in `BaseLoraTabView.build_lora_options`.
5. Export: check `convert_lora_util.py` for key-suffix handling per `ModelFormat`, and `_check_rank_matches` rank-key map (currently disabled, FIXME at `load_state_dict`).
6. The LoRA `setup_model` in every `modelSetup/*LoRASetup.py` goes through `LoRAModuleWrapper`, so no per-model change is normally needed.

### New model type
1. `ModelType` enum: the value, an `is_x()`, and entries in `is_flow_matching`, `supported_training_methods`, `supported_lora_formats`, `supported_full_model_formats` and **`_MODEL_PARTS`** (a missing entry raises KeyError).
2. `modules/model/XModel.py` (a `BaseModel` subclass).
3. Loader: `modelLoader/x/XModelLoader.py` (+ `XLoRALoader`, `XEmbeddingLoader`), plus top-level `XFineTuneModelLoader.py` / `XLoRAModelLoader.py` using `make_*_model_loader`.
4. Saver: the same pattern under `modelSaver/x/` plus top-level `make_*_model_saver`.
5. Setup: `modelSetup/BaseXSetup.py` (mixins, `LAYER_PRESETS`, `predict`, `calculate_loss`), plus `XLoRASetup`/`XFineTuneSetup` with `@factory.register`.
6. `dataLoader/XBaseDataLoader.py` (`@factory.register(BaseDataLoader, ModelType.X)`) and `modelSampler/XSampler.py`.
7. `modules/util/checkpointing_util.py`: `enable_checkpointing_for_x_transformer`.
8. `resources/sd_model_spec/x.json`, `x-lora.json`; `training_presets/X/#*.json`.
9. UI and config touchpoints (grep `is_chroma()` to find the full list): `TopBarController.get_model_types`, `ModelTabController`, `BaseTrainingTabView` (dispatch at top), `SampleConfig.default_values`, `BaseConvertModelUIView`, `modules/util/optimizer/muon_util.py`.

### New config field (TrainConfig or a sub-config)
1. Add a class annotation **and** a `data.append((name, default, type, nullable))` in `default_values()`. Only fields in `default_values()` get (de)serialized; an annotation alone does nothing.
2. Supported types: `str/int/float/bool`, `Enum`, a nested `BaseConfig`, and `list`/`dict` of those (`BaseConfig.to_dict/from_dict`).
3. Renaming, moving or changing meaning: bump `config_version` (currently 11, `TrainConfig.py:596`) and add `__migration_N`. **Also edit the built-in `training_presets/**/#*.json` by hand**, because they load with `migrate=False`.
4. UI: add a widget in the matching `modules/ui/Base*View.py` (shared by Qt and CTk) bound by name: `self.components.entry(frame, row, col, ui_state, "field")` or `"transformer.field"`. Choices come from the matching `*Controller`. A new widget kind has to be added to **both** `modules/util/ui/pyside6_components.py` and `ctk_components.py` (same function names).
5. Consume it in `modelSetup`/`create.py`/`dataLoader`. If it changes cached data, add it to the data loader's cache split names.
6. Optimizer fields: see "New optimizer" step 4. Per-part fields go on `TrainModelPartConfig`; the migrations fan per-part values out over all 12 part names.

## Conventions
- Naming: `<Model><Method>{ModelLoader,ModelSaver,Setup}.py`, `Base<Model>Setup.py`, `<Model>BaseDataLoader.py`, `<Model>Sampler.py`. Model-specific helpers go in lowercase subfolders (`modelLoader/chroma/`).
- Config: every config is a `BaseConfig` built from `default_values()`. Enums serialize as their string value (each enum defines `__str__`). `None` means "use the fallback in the consumer" for many optimizer fields.
- Setup classes are built from mixins (`modelSetup/mixin/`, `dataLoader/mixin/`, `modelLoader/mixin/`). Override the hook; don't copy the pipeline.
- UI = `Base*View` (toolkit-agnostic, gets a `components` module) + `Ctk*View`/`PySide6*View` (layout only) + `*Controller` (logic). `UIState` (`modules/util/ui/UIState.py`) writes widget changes straight into the config object. `""`/`"None"` become `None` only for nullable fields.
- Imports follow the ruff isort sections: future, stdlib, `modules`, `mgds`, `torch`, `diffusers/transformers`, third-party. Long tables are wrapped in `# @formatter:off/on`.
- Scripts are thin: `script_imports()` first, then logic lives in `modules/`.

## Gotchas
**Config loading**
- `BaseConfig.from_dict` (`:66`) **never fails**. A missing key keeps its default. A bad value prints `Could not set X as Y` (`:134`) and is skipped. Unknown keys (e.g. `weight_dtype` in the Chroma presets) are ignored silently.
- Loading a preset in the UI applies it over **defaults**, not over the current config (`TopBarController.load_config_from_file:83`). A filename starting with `#` (except `#.json`) is treated as built-in and **skips migration**.
- The CLI never applies `OPTIMIZER_DEFAULT_PARAMETERS`; only the UI (`change_optimizer`) does. A hand-written CLI config therefore gets the inline fallbacks in `create.py`, which can differ. Example: `ADOPT_ADV`/`ADAMW_ADV`/`PRODIGY_ADV` `beta1` falls back to **0** on the CLI but defaults to 0.9 in the UI. Set optimizer params explicitly in CLI configs.
- Saved configs (`to_settings_dict`) point to `concept_file_name`/`sample_definition_file_name` by path. Exported "pack" configs (`to_pack_dict`) inline them. `samples.json` is re-read at every sample.
- `layer_filter_preset` exists only for the UI. Training reads only `layer_filter` (comma-separated substrings, or regex when `layer_filter_regex`). In LoRA setups it applies to the denoiser wrapper only, not the TE LoRA. A filter that matches nothing raises `ValueError`.

**Optimizers / schedulers**
- `is_schedule_free` optimizers (SCHEDULE_FREE_ADAMW/SGD, PRODIGY_PLUS_SCHEDULE_FREE) are **silently forced** to a CONSTANT scheduler with no warmup (`create.py:1137`, `:1222`). PRODIGY_ADV is *not* flagged schedule-free.
- `learning_rate_warmup_steps` (default **200**): >1 means steps (divided by grad-accum), 0<x≤1 means a *fraction* of total steps (so `1` means warmup across the whole run), ≤0 means none (`create.py:1127`).
- The ADV optimizers get `k_warmup_steps = learning_rate_warmup_steps / grad_accum` for Kourkoutas-β (`create.py:689,714,746`). With LR warmup at 0, that warmup is 0 too.
- `ADAMW_ADV`/`ADOPT_ADV` read `beta3_ema` from `optimizer_config.beta3` (`create.py:686,709`), but the UI edits `beta3_ema`. The UI "Beta3 EMA" value is **ignored** (always 0.9999) for those two. PRODIGY_ADV reads the right field.
- bitsandbytes backs every `*_8BIT` optimizer plus ADAGRAD, RMSPROP, LARS, LAMB and AdEMAMix (even 32-bit), and the `INT_8`/`NFLOAT_4` weight dtypes. `*_COMPRESSED` dtypes need nvCOMP (NVIDIA only) and raise otherwise.
- A layer-offloaded part in FINE_TUNE requires an optimizer with `supports_fused_back_pass()` **and** `fused_back_pass=true` (`create.py:141`).

**Adapters**
- DoRA is `lora_decompose` for LoRA and `lokr_weight_decompose` for LoKr. LoHa and OFT v2 have no DoRA path; `lora_decompose` is ignored for them.
- LoKr uses `lokr_dim` (not `lora_rank`) and `lora_alpha`. If `lokr_dim >= max(w2 dims)/2`, it prints "using full matrix mode" and makes W2 full (`LoRAModule.py:365`). If W1 and W2 both end up full, **alpha is overwritten with `lokr_dim`** (scale 1, `:445`), so `lora_alpha` is ignored.
- `lokr_decompose_factor=-1` (the default) factorizes near √dim; for example 3072 → 48×64.
- The output format changes the adapter's structure. `ORIGINAL_LORA/COMFY_LORA/KOHYA_LORA` build **fused qkv** adapters (`ModelFormat.needs_qkv_fusion`), while DIFFUSERS/LEGACY keep them split. Resuming a LoRA or backup under a different fusion mode fails `check_fusion_match`.
- `DIFFUSERS_LORA` (the Chroma LoRA presets' format) with per-input DoRA (`*_dora_on_output`/`lora_decompose_output_axis` off) stores `dora_scale` that diffusers drops. LoHa/LoKr/OFT in DIFFUSERS format can't be loaded by HF diffusers (`convert_lora_util.py:162`).

**Data / masking**
- `unmasked_probability` works **only** for inpainting models (`has_mask_input()`); for Chroma and every other txt2img model it does nothing (`DataLoaderText2ImageMixin.py:312`). For those, masked training uses `unmasked_weight` + `normalize_masked_area_loss`.
- `masked_prior_preservation_weight` only takes effect for `TrainingMethod.LORA` (`GenericTrainer.py:744`).
- Toggling `masked_training`, `latent_caching` or TE training changes what gets cached. `clear_cache_before_training` defaults to true (the UI asks for confirmation). If you turn it off, clear `cache_dir` yourself after changing those settings (the full list of cache-key inputs is **unverified**; it lives in mgds `DiskCache`).
- `dataloader_threads > 1` together with a text-encoder `offload_fraction > 0` raises an error.

**EMA / saving**
- Intermediate saves and the final model contain **EMA weights** when EMA is on (`GenericTrainer.__save`, `end`). Backups keep raw weights plus EMA state.
- EMA updates run only every `ema_update_step_interval` (default 5) optimizer steps but use the per-step decay, so the effective horizon is about `interval/(1-decay)` steps (≈5000 at defaults). Decay ramps as `(1+s)/(10+s)`. With EMA on, `non_ema_sampling=true` (default) doubles the sampling cost.
- Ctrl-C in `scripts/train.py` saves only if `backup_before_save` is true, and `end()` saves nothing unless at least one optimizer step ran.
- `update.sh`/`install.sh` run `pip --upgrade-strategy eager -r requirements-rocm.txt`, which **reinstalls the pinned torch** (`2.12.0+rocm7.2` today) over any manually installed build.

**Attention**
- `attention_mechanism`: `SDP` = diffusers "native" (torch SDPA picks its own kernel); `FLASH` = diffusers "flash" (the flash-attn package, **CUDA-oriented**); `CUDNN` = **CUDA-only**; `FLEX` = torch FlexAttention. Chroma always passes a text attention mask when captions are padded (`BaseChromaSetup.predict`).

## Working rules for Claude
- Don't change default config values, optimizer defaults, loss/noise/timestep math, or training-loop order without asking first.
- Keep diffs minimal and scoped. No drive-by refactors or reformatting (the repo uses ruff lint only, not ruff format).
- Flag every change that would break existing preset JSONs, saved `training_configs/`, `training_presets/#.json`, backups, or saved LoRA key layouts. Renames need a migration **and** hand-edited built-in presets.
- Run `ruff check` on touched files. There are no tests, so say plainly what was not exercised by a real run.
- Treat anything CUDA-only (bitsandbytes CUDA kernels, xformers, flash-attn, cuDNN attention, nvCOMP) as unavailable here: flag it and offer a ROCm alternative.

## My environment (user-provided; not derived from the repo)
- Fedora Linux; AMD RX 7900 XTX (gfx1100 / RDNA3, 24 GB VRAM); Ryzen 9 7950X3D; 128 GB RAM.
- ROCm 7.2.4 is the pinned, working stack. The upgrade path is torch 2.13.0+rocm7.2.
  - Do NOT suggest ROCm 10.0 / TheRock 7.14 builds. torch 2.12.0+rocm7.14.0 has an AOTriton packaging gap that wrongly enables flash SDPA and causes NaNs.
  - Note: the repo's `requirements-rocm.txt` pins `torch==2.12.0+rocm7.2`, and `update.sh` will reinstall it (see Gotchas).
- Never suggest CUDA-only fixes (bitsandbytes CUDA paths, xformers, CUDA flash-attn) without flagging them as CUDA-only and offering a ROCm alternative.
- This build supports the `*_ADV` optimizers (ADOPT_ADV, PRODIGY_ADV, ADAMW_ADV). **ADOPT_ADV is the current best performer.** Prodigy-plus-schedule-free needs a CONSTANT scheduler (the code also forces it).
- DoRA stacks with LoRA and LoKr here, but not with LoHa or OFT v2. With LoKr, `lokr_decompose_factor=-1` falls back to the full matrix at dim 32 on 3072-dim weights, so **use factor 16**.
- Main workload: **Chroma1-HD** (`CHROMA_1`) at 512 px, masked training, EMA on, no LR warmup (set `learning_rate_warmup_steps: 0`, since the default is 200).
