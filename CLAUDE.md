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
./run-cmd.sh generate_debug_report   # includes a "PyTorch GPU Runtime" section (HIP/CUDA build, gfx arch, VRAM, bnb library, /opt/rocm version)
./run-cmd.sh rocm_check [--full]    # GPU check suite (docs/GpuCheck.md): kernel correctness, training code, speed per model family
ruff check .                     # lint config in pyproject.toml; E501 ignored, line-length 120
pre-commit run --all-files       # pre-commit hooks + ruff --fix (linter only, NOT ruff format)
```
- **No test suite.** No GitHub Actions workflows; only pre-commit.ci. Validate changes with `ruff check` plus a short real run (e.g. `--config-value epochs=1`).
- `scripts/rocm_check.py` (logic in `modules/util/rocm_check/`) is the only GPU-side check: the user runs it on the 7900 XTX and hands back `rocm_check_report.txt`/`.json`. Model shapes live in `shapes.py` (one row per family, not per model), so a new model type with new head/hidden sizes adds a row there. On the CPU, GPU-only checks skip; everything else runs (offloading uses `cpu:1` as the temp device).
- UI checks without a display: build `PySide6TrainView` with `QT_QPA_PLATFORM=offscreen` (needs `libEGL.so.1`, `libxkbcommon`, `libfontconfig`) and use `widget.grab().save(...)`. Before `findChildren`, flush `deleteLater()`d widgets with `QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)`, because `processEvents()` leaves them in place. The CTk UI needs a Python with tkinter plus Xvfb.
- `--config-value KEY=VALUE`: dotted keys walk nested `BaseConfig`s (`optimizer.beta1=0.9`); `None`/`null` clears a nullable field. `ema` is an `EMAMode` enum; the decay field is `ema_decay`.
- `--preset-path` turns off migration for **both** files, so the config must already be in the current format (`__version` 12).
- Python: min 3.10, <3.14; conda path uses 3.13 (`lib.include.sh:35-37`). `OT_PREFER_VENV` defaults to `true` (Venv even if Conda is installed); set it to `false` to use Conda.

## Architecture (one training run)
```
scripts/train.py:main                      (CLI)          scripts/train_ui_qt.py -> modules/ui/PySide6TrainUIView.py (UI)
  TrainConfig.default_values()                             TrainUIController.start_training:282 writes training_presets/#.json
  .from_dict(preset, migrate=False)                          + secrets.json, trains in a thread on the *live* config object
  .from_dict(config, migrate=preset is None)
  --config-value overrides; secrets.json -> config.secrets
  -> modules/util/create.py:create_trainer:1486  (cloud.enabled -> CloudTrainer | multi_gpu -> MultiTrainer | GenericTrainer)

modules/trainer/GenericTrainer.py
  start():88
    create_model_loader  -> factory.get(BaseModelLoader, model_type, training_method)
                            e.g. modelLoader/ChromaLoRAModelLoader.py (make_lora_model_loader) -> modelLoader/chroma/*
    model_setup.setup_optimizations  -> per-part quantize / checkpointing / offload, attention backend (BaseModelSetup._setup_model_part)
    model_setup.setup_train_device
    model_setup.setup_model          -> LoRAModuleWrapper(...) per part, load lora_state_dict, hook_to_module()
                                        -> optimizer_util.init_model_parameters -> create.create_optimizer:207 + create_ema:1200
    create_data_loader   -> dataLoader/<Model>BaseDataLoader.py (registered per ModelType only)
                            DataLoaderText2ImageMixin._create_dataset:375 -> DataLoaderMgdsMixin._create_mgds -> mgds.MGDS/TrainDataLoader
    create_model_saver / create_model_sampler
  train():616   per epoch: dataset.start_next_epoch() (does caching) -> lr scheduler created lazily (create_lr_scheduler:1225)
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
1. `modules/util/enum/Optimizer.py`: add the member. Update `is_adaptive`, `is_schedule_free`, `supports_fused_back_pass()` and `is_adv_optm` (adv_optm classes: resume version gate) if they apply.
2. `modules/util/create.py:create_optimizer`: add a `case`. The `match` has **no default**, so an unhandled enum gives `optimizer=None`.
3. `modules/util/optimizer_util.py:OPTIMIZER_DEFAULT_PARAMETERS[Optimizer.X]` is **required**; `change_optimizer` raises KeyError when the optimizer is picked in the UI. Its keys decide which params the UI shows.
4. A new hyperparameter needs: a `TrainOptimizerConfig` annotation plus a `default_values()` entry (`TrainConfig.py:35`/`:146`), a `KEY_DETAIL_MAP` entry in `modules/ui/BaseOptimizerParamsWindowView.py:33` (keys missing there are silently **hidden**), and use in `create.py`.
5. Pin the package in `requirements-global.txt`, or in the platform files if it has GPU builds. If it is CUDA-only, say so. bitsandbytes is pinned per platform file, currently 0.49.2 everywhere (the first 0.49.x with a `rocm72` binary; its CUDA wheels ship cuda126 and cuda130 on Linux, cuda118–130 on Windows). 0.50 removed the `percentile_clipping`/`block_wise` constructor args that `create.py` passes to ADAM(W)_8BIT, ADAGRAD_8BIT, RMSPROP_8BIT, LARS_8BIT and LAMB(_8BIT) (TypeError), so an upgrade needs those removed first.

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
3. Renaming, moving or changing meaning: bump `config_version` (currently 12, `TrainConfig.py:614`) and add `__migration_N`. **Also edit the built-in `training_presets/**/#*.json` by hand**, because they load with `migrate=False`.
4. UI: add a widget in the matching `modules/ui/Base*View.py` (shared by Qt and CTk) bound by name: `self.components.entry(frame, row, col, ui_state, "field")` or `"transformer.field"`. Choices come from the matching `*Controller`. A new widget kind has to be added to **both** `modules/util/ui/pyside6_components.py` and `ctk_components.py` (same function names).
5. Consume it in `modelSetup`/`create.py`/`dataLoader`. If it changes cached data, add it to the data loader's cache split names.
6. Optimizer fields: see "New optimizer" step 4. Per-part fields go on `TrainModelPartConfig`; the migrations fan per-part values out over all 12 part names.

## Conventions
- Naming: `<Model><Method>{ModelLoader,ModelSaver,Setup}.py`, `Base<Model>Setup.py`, `<Model>BaseDataLoader.py`, `<Model>Sampler.py`. Model-specific helpers go in lowercase subfolders (`modelLoader/chroma/`).
- Config: every config is a `BaseConfig` built from `default_values()`. Enums serialize as their string value (each enum defines `__str__`). `None` means "use the fallback in the consumer" for many optimizer fields.
- Setup classes are built from mixins (`modelSetup/mixin/`, `dataLoader/mixin/`, `modelLoader/mixin/`). Override the hook; don't copy the pipeline.
- UI = `Base*View` (toolkit-agnostic, gets a `components` module) + `Ctk*View`/`PySide6*View` (layout only) + `*Controller` (logic). `UIState` (`modules/util/ui/UIState.py`) writes widget changes straight into the config object. `""`/`"None"` become `None` only for nullable fields.
- UI preferences (settings tab: theme, UI scale, Qt font size, window size, Linux system file dialogs) are a separate `UISettingsConfig` in `training_user_settings/ui_settings.json` (gitignored), not part of `TrainConfig`. With no file, both UIs start **dark**. Qt applies the scale only at startup (`create_application` sets `QT_SCALE_FACTOR` unless it is already in the environment). CTk applies the theme only at startup, because labels in its transparent scrollable tabs keep their old background after a live switch.
- Qt: `QApplication.setFont` does not reach widgets already polished by the application stylesheet; re-set the stylesheet afterwards (`pyside6_util.apply_font_size`). Programmatic `var.set()` on a `QtVar` runs widget callbacks (e.g. a dropdown's `command`) **before** the trace that writes the config, so write the config first when a callback reads it (`LoraTabController.activate`).
- Qt widgets (and `QPixmap`) may only be touched on the UI thread. Worker threads (concept stats scan, sampling, video tool, training callbacks) hand UI work over with `components.call_after(widget, 0, fn)` or `schedule_on_main_thread` (`QTimer.singleShot(0, context, fn)` runs `fn` in the context's thread). A direct call from a thread may freeze or crash the app on a real desktop, even when offscreen tests pass.
- Qt file dialogs go through `pyside6_util.file_dialog_options()`: on Linux it adds `DontUseNativeDialog` (the desktop's GTK/portal/KDE dialog loaded into pip Qt can freeze or crash the app) unless Settings > System File Dialogs is on.
- Offscreen Qt runs no popups or native dialogs. For those, run under Xvfb with `QT_QPA_PLATFORM=xcb` (needs `libxcb-cursor0`, `libxkbcommon-x11-0`, `libxcb-icccm4`, `libxcb-keysyms1`, `libxcb-shape0`, `libxcb-xkb1`); `QT_QPA_PLATFORMTHEME=gtk3` or `xdgdesktopportal` picks the native dialog backend.
- Imports follow the ruff isort sections: future, stdlib, `modules`, `mgds`, `torch`, `diffusers/transformers`, third-party. Long tables are wrapped in `# @formatter:off/on`.
- Scripts are thin: `script_imports()` first, then logic lives in `modules/`.

## Gotchas
**Config loading**
- `BaseConfig.from_dict` (`:66`) **never fails**. A missing key keeps its default. A bad value prints `Could not set X as Y` (`:134`) and is skipped. Unknown keys (e.g. `weight_dtype` in the Chroma presets) are ignored silently.
- Loading a preset in the UI applies it over **defaults**, not over the current config (`TopBarController.load_config_from_file`). A filename starting with `#` (except `#.json`) is treated as built-in and **skips migration**. Like `scripts/train.py`, the load goes on top of `default_optimizer_config(optimizer)`, and the loaded optimizer settings become that optimizer's `optimizer_defaults` entry (they used to be overwritten by it).
- `optimizer_defaults` remembers each optimizer's settings for switching back to it; the optimizer window saves into it through `OptimizerParamsWindowController.on_close`, which every widget command calls. It skips saving while it loads an optimizer's settings into the window (a save then stored a half-updated mix).
- The concepts/samples tabs pick their file through a dropdown of `training_concepts/`/`training_samples/`. The file a config names is always added to it (`BaseConfigListView.__list_config_file`), and `reload_from_config()` runs after every config load: before, a file outside those folders left the previous list shown, and saving wrote it over the loaded config's file.
- Switching the model type sets `base_model_name` and the other parts' `model_name` (decoder, effnet, HiDream's Llama) from the built-in presets, or from `TopBarController.FALLBACK_BASE_MODELS` for types without a preset, but only while the field is empty or holds another type's default.
- Input validation (`util/ui/validation.py`): floats must be non-negative except `SIGNED_FLOAT_FIELDS` (noising bias/weight, Adafactor's `decay_rate`), and an empty non-nullable int/float is "Value required" (it used to be stored as `None`).
- The UI restores the last session from `training_presets/#.json` (gitignored). Without it (fresh clone or worktree) the UI starts as SD1.5 / Fine Tune, which has no `blocks` layer-filter preset (only transformer models define one). The LoRA, embedding and additional embeddings tabs are always shown. LoRA/embedding are greyed out with a "Switch to ..." button unless their method is selected, or show a "not supported" note for models without it (`util/ui/training_method_notice.py`); additional embeddings are greyed out with a note on models without embedding training (`BaseAdditionalEmbeddingsTabView.update_supported`, re-applied after every list change). If `#.json` fails to load, it is copied to `#.json.bak` first, because closing the UI overwrites `#.json` with the defaults.
- Every optimizer gets the UI's `OPTIMIZER_DEFAULT_PARAMETERS` on the CLI too: `scripts/train.py` loads the config on top of `default_optimizer_config()`, and `create_optimizer` fills unset (None) settings from the table, except where None is itself a setting (`optimizer_util._NONE_IS_A_SETTING`, e.g. prodigy-plus `eps=None` = Adam-atan2). Explicit non-null values always win, so a `create_train_files` template switched to another optimizer keeps its written bools (e.g. ADOPT_ADV `use_atan2: false`, UI default true).
- `output_model_format` defaults to `SAFETENSORS`, which saves only embeddings. The UI's Output Format dropdown switches to the first valid format, but a CLI config (e.g. the `create_train_files` template) has to set one from `ModelType.supported_output_formats(training_method)`. `GenericTrainer.start` raises otherwise; before that check, the saver failed at the first save, after the training.
- LARS, LARS_8BIT, SGD_8BIT and SCHEDULE_FREE_SGD reject `momentum: 0`, and DADAPT_ADA_GRAD rejects `eps: 0.0` (defaults are now 0.9 / 1e-6). Configs saved with the old values still fail until the value is changed.
- A config file without `"__version"` is migrated from version 0, and `__migration_0` crashes (`unhashable type: 'dict'`) if `optimizer` is a nested object. Always keep `__version`.
- Saved configs (`to_settings_dict`) point to `concept_file_name`/`sample_definition_file_name` by path. Exported "pack" configs (`to_pack_dict`) inline them. `samples.json` is re-read at every sample.
- `layer_filter_preset` exists only for the UI. Training reads only `layer_filter` (comma-separated substrings, or regex when `layer_filter_regex`). In LoRA setups it applies to the denoiser wrapper only, not the TE LoRA. A filter that matches nothing raises `ValueError`.

**Optimizers / schedulers**
- `is_schedule_free` optimizers (SCHEDULE_FREE_ADAMW/SGD, PRODIGY_PLUS_SCHEDULE_FREE) are **silently forced** to a CONSTANT scheduler with no warmup (`create.py:1251`, `:1336`). PRODIGY_ADV is *not* flagged schedule-free.
- `learning_rate_warmup_steps` (default **200**): >1 means steps (divided by grad-accum), 0<x≤1 means a *fraction* of total steps (so `1` means warmup across the whole run), ≤0 means none (`create.py:1241`).
- The ADV optimizers get `k_warmup_steps = learning_rate_warmup_steps / grad_accum` for Kourkoutas-β (`create.py:768,788,815`). With LR warmup at 0, that warmup is 0 too.
- `ADAM_8BIT`/`ADAMW_8BIT` use `bnb.optim.Adam8bit`/`AdamW8bit` (always 8-bit state; tensors < `min_8bit_size` stay fp32; `amsgrad` unsupported). bnb picks the update kernel from the stored state dtype, so old 32-bit state from a backup keeps working. Other bnb optimizers still show UI params that `create.py` never passes (e.g. `optim_bits`/`min_8bit_size`/`percentile_clipping` for ADAGRAD, RMSPROP, LARS; `block_wise` etc. for SGD_8BIT); their displayed defaults match the actual behavior.
- bitsandbytes backs every `*_8BIT` optimizer plus ADAGRAD, RMSPROP, LARS, LAMB and AdEMAMix (even 32-bit), and the `INT_8`/`NFLOAT_4` weight dtypes. `*_COMPRESSED` dtypes need nvCOMP (NVIDIA only) and raise otherwise.
- The float W8A8 dtypes (`FLOAT_W8A8`, its compressed variant, `GGUF_A8_FLOAT`) run `torch._scaled_mm`, which PyTorch allows only on fp8 hardware (NVIDIA sm89+, AMD gfx942/gfx950/gfx1200/gfx1201). `GenericTrainer.start` probes it (`torch_util.fp8_matmul_supported`) and stops before loading; it used to fail at the first forward, after caching. On gfx1100 (RDNA3), int8 goes to native WMMA (hipBLASLt ships navi31 I8II kernels for `_int_mm`; Triton emits `v_wmma_i32_16x16x16_iu8`) and fp8 only as an fp16 upcast, and RDNA3 runs int8 WMMA at the bf16 rate: W8A8 saves memory but is not faster there, same as `FLOAT_8` (W8, weight-only). The W8A8 backward is a Triton kernel, so W8A8 can't train on the CPU. Its first use autotunes 16 variants: for gfx1100 the int8 ones compile in ~3 min in total (5 of them need more than the 64 KiB LDS and are skipped at launch), but the fp8 ones (emulated via fp16) took >10 min each, which is why `rocm_check` tests fp8 only where `fp8_matmul_supported`.
- A layer-offloaded part in FINE_TUNE requires an optimizer with `supports_fused_back_pass()` **and** `fused_back_pass=true` (`create.py:225`).

**adv_optm (the `*_ADV` optimizers, pinned 2.5.13)**
- 2.5 removed AdEMAMix / Simplified AdEMAMix (`use_AdEMAMix`, `beta3_ema`, `alpha_grad`, `Simplified_AdEMAMix`), Lion's `clip_threshold` and `Lion_Prodigy_adv`. Their successor is `nesterov` + `nesterov_coef` (update = coef·momentum + (1−coef)·grad; `None` = beta1/momentum; needs beta1/momentum > 0). The bitsandbytes `ADEMAMIX`/`ADEMAMIX_8BIT` optimizers are separate and unchanged, and so is the shared `alpha` field.
- `orthogonal_gradient` is a mode string (`disabled`/`flattened`/`iterative`), not a bool: adv_optm returns `None` for a bool mode, which nulls every gradient. `create._adv_orthograd_mode` maps a leftover bool (also in the Muon aux-Adam dict). `nnmf_factor` became `state_precision="factored"` except on LION_ADV, which has no `state_precision`. `__migration_11` does these renames.
- `state_precision` values that work are listed by `optimizer_util.adv_state_precisions` (UI dropdown and a `ValueError` in `create.py`): `fp16` breaks everything but ADOPT_ADV, `factored` breaks ADAMUON_ADV, and SINKSGD_ADV `factored` with momentum 0 is passed as `auto` (no state to factor). Muon/AdaMuon without the aux Adam crash on 1D params (biases, norms) in both 2.2.3 and 2.5.13; LoRA params are 2D.
- Resuming: optimizer.pt carries `adv_optm_version`. A different major.minor (or none, i.e. ≤2.2) drops the optimizer state with a warning; weights/EMA/progress still resume. On load, `create.py` also works around two adv_optm 2.5.13 bugs: it sets `actual_state_precision` on groups that lack it (Lion, all-factored), and `_restore_adv_optm_state_dtypes` puts the saved state tensors back after `load_state_dict`, because torch's per-param cast plus adv_optm's re-cast broke `int8_sr` (signed → uint8), fp32 state on bf16 weights, and fp32 vector state in `factored` mode.
- Muon groups are saved as `<name>_<optim_type>`; `create_optimizer` maps them back whenever any group has an `optim_type` (before this, Muon without the aux Adam silently resumed with a fresh state).
- `centered_wd` > 0 stores an anchor copy of the weights (`centered_wd_mode`, default float8 e4m3, also for LoRA params). float8 is a storage-only cast; untested on ROCm.

**Adapters**
- DoRA is `lora_decompose` for LoRA and `lokr_weight_decompose` for LoKr. LoHa and OFT v2 have no DoRA path; `lora_decompose` is ignored for them.
- Conv LoHa factors are saved in the LyCORIS 2D layout (`hada_w*_a` [out, rank], `hada_w*_b` [rank, in·k·k]); `LoHaModule._load_from_state_dict` also accepts the older 4D layout.
- LoKr uses `lokr_dim` (not `lora_rank`) and `lora_alpha`. If `lokr_dim >= max(w2 dims)/2`, it prints "using full matrix mode" and makes W2 full (`LoRAModule.py:365`). If W1 and W2 both end up full, **alpha is overwritten with `lokr_dim`** (scale 1, `:445`), so `lora_alpha` is ignored.
- `lokr_decompose_factor=-1` (the default) factorizes near √dim; for example 3072 → 48×64.
- The output format changes the adapter's structure. `ORIGINAL_LORA/COMFY_LORA/KOHYA_LORA` build **fused qkv** adapters (`ModelFormat.needs_qkv_fusion`), while DIFFUSERS/LEGACY keep them split. Resuming a LoRA or backup under a different fusion mode fails `check_fusion_match`.
- `DIFFUSERS_LORA` (the Chroma LoRA presets' format) with per-input DoRA (`*_dora_on_output`/`lora_decompose_output_axis` off) stores `dora_scale` that diffusers drops. LoHa/LoKr/OFT in DIFFUSERS format can't be loaded by HF diffusers (`convert_lora_util.py:162`).

**Data / masking**
- `unmasked_probability` works **only** for inpainting models (`has_mask_input()`); for Chroma and every other txt2img model it does nothing (`DataLoaderText2ImageMixin.py:312`). For those, masked training uses `unmasked_weight` + `normalize_masked_area_loss`.
- `masked_prior_preservation_weight` only takes effect for `TrainingMethod.LORA` (`GenericTrainer.py:744`).
- mgds `DiskCache` keys the cache only by the concept (path, seed, subdirectories flag, image/text settings), not by model, resolution, dtypes or masking. With `clear_cache_before_training` off, those changes used to reuse stale latents silently (e.g. 128 px latents for a 256 px run). `GenericTrainer.start` now keeps `cache_settings.json` in `cache_dir` (`util/cache_util.py`) and rebuilds the cache when a recorded setting changed or the record is missing. A concept without a `seed` gets a random one per run, so its cache is never reused anyway.
- `dataloader_threads > 1` together with a text-encoder `offload_fraction > 0` raises an error.
- Aspect batch sorting never mixes buckets and skips each bucket's leftover samples, so a small dataset can give an epoch with no batches. `GenericTrainer.train` then raises instead of counting through empty epochs (which would end without saving).
- Caching runs the encoders (VAE, text encoders) in `dataloader_threads` concurrent threads (mgds `PipelineState` executor), so with the default 2, two samples are prepared at once. A `CheckFinite` module (`dataLoader/pipelineModules/`) in front of each disk cache stops caching with the file name when an encoding has NaN/inf. Seen on ROCm (gfx1100, torch 2.13+rocm7.2): a random image's latent came out all NaN, and the same image encoded fine on a rerun. The cause turned out to be the AOTriton head-size-512 bug in the VAE's attention (see Attention), not concurrency; the shared lock the model-holding pipeline modules take (`DataLoaderMgdsMixin._run_models_one_at_a_time`) stays, so encodes run one at a time while loading/cropping stays parallel, and the cache is bit-identical to `dataloader_threads: 1`.
- The NaN-loss check runs **before** `optimizer.step()` (`GenericTrainer.train`), so a NaN batch never reaches the weights or optimizer state; with a fused back pass the update already happened during backward. It also stops on an infinite loss (which broke the weights the same way), except with an enabled fp16 grad scaler, which skips such steps itself. `CustomGradScaler` is built for device `cuda`, so it is disabled (but not `None`) on a CPU run.
- Validation (`GenericTrainer.__validate`) runs like sampling: model in eval mode (no LoRA/T5 dropout), schedule-free optimizers switched to their averaged weights (`__before_eval`), then `setup_train_device` and `optimizer.train()` restore training. It measures the raw weights, not EMA.

**Layer / activation offloading** (`modules/util/LayerOffloadConductor.py`)
- Async transfers use three streams (train = default, layer, activations) and are on whenever `train_device` is `cuda` (ROCm too) and `async_offloading` is on. Temp-side caches are pinned with `cudaHostRegister`/`hipHostRegister` only in async mode; `async_offloading: false` = synchronous copies, no pinned memory.
- Stream-ordering invariants (fixed bugs, keep them): a layer offloaded during the **back pass** must wait for all train work queued so far, because `after_layer()` runs at the end of the checkpoint recompute, before that layer's backward kernels (which read the weights; fused back pass also updates them) are queued. Pinned buffers must only be unpinned after a device sync (`unpin_tensors_`).
- Offloadable parts: every transformer, T5/LLM/CLIP text encoder, the SD/SDXL UNet (its transformer blocks, in down→mid→up order; `model.modules()` lists up before mid), and the Wuerstchen/Stable Cascade prior (a Cascade UNet with repeating blocks is rejected). Not: VAE, effnet, decoder. The conductor is stored as `model.<attr>_offload_conductor`; `BaseModel._move_part` finds it by name. A layer list must be in exact execution order, one call per layer per step.
- When a part offloads, diffusers' own `gradient_checkpointing` must be off on the modules whose blocks OneTrainer wraps (Transformer2DModel, the priors): a nested checkpoint re-runs the offload forward in the middle of the back pass. Without offloading it is left as it was.
- `OffloadCheckpointLayer` turns kwargs into positional args for the reentrant checkpoint; arguments only a `**kwargs` parameter takes are passed by name (CLIP's `is_causal=True`; OneTrainer calls CLIP without a mask, so dropping it meant non-causal attention). A grad-requiring tensor there raises.
- LLM text encoders must be called with `use_cache=False` when grad can flow through them: a checkpoint recompute appends to the KV cache again. Qwen/Z-Image/Anima/Krea2/Ideogram don't pass it, which is fine only because their encoder never runs with grad.
- Activation offloading swaps the listed block input's storage in place (`tensor.data = ...`), so it is only safe for inputs that nothing outside the checkpointed block saves for backward.
- Offloading needs `gradient_checkpointing` (use_reentrant=True recompute drives `before_layer`/`after_layer` in backward). Only `nn.Linear`/`Conv2d` weights+bias (and SVD/NF4 parts) are offloaded (`quantization_util.get_offload_tensors`).
- Test offloading on CPU by driving the real conductor with `train_device="cpu"`, `temp_device="cpu:1"` (different to `device_equals`, same memory).
- `materialize()` copies each layer's offloadable tensors straight into the static cache of the device the layer starts on and moves only the rest of the layer to the train device (`__layer_to_device`); it used to move every layer whole to the train device first (offloaded ones there and back).

**GPU memory**
- `BaseModel.materialize/evict/materialize_only` is the only way parts reach the train device, and `materialize_only` evicts before it loads, so two large parts (e.g. Chroma's 17.8 GB bf16 transformer and 9.5 GB T5) are never resident together. Sampling, caching and backups only swap between optimizer steps, after `zero_grad(set_to_none=True)`.
- `OT_LOG_VRAM=true` prints tensor/reserved/free GPU memory after every load and unload; an OOM while loading a part prints which part and the same numbers (`torch_util.vram_summary`).
- `OT_EXPANDABLE_SEGMENTS=true` (launcher) adds `expandable_segments:True` to `PYTORCH_CUDA_ALLOC_CONF`, merged with `OT_CUDA_LOWMEM_MODE`'s settings: PyTorch reads only the first of `PYTORCH_CUDA_ALLOC_CONF`, `PYTORCH_HIP_ALLOC_CONF`, `PYTORCH_ALLOC_CONF`. Off by default, because the allocator has no fallback when the device lacks virtual memory support (every allocation then fails). On the user's 7900 XTX it works: in `rocm_check`'s churn pattern (varying sizes, then one large block) it held 1.65 GiB reserved-but-unused vs 12.8 GiB with the default allocator.

**EMA / saving**
- Intermediate saves and the final model contain **EMA weights** when EMA is on (`GenericTrainer.__save`, `end`). Backups keep raw weights plus EMA state.
- EMA updates run only every `ema_update_step_interval` (default 5) optimizer steps but use the per-step decay, so the effective horizon is about `interval/(1-decay)` steps (≈5000 at defaults). Decay ramps as `(1+s)/(10+s)`. With EMA on, `non_ema_sampling=true` (default) doubles the sampling cost.
- Ctrl-C in `scripts/train.py` saves only if `backup_before_save` is true, and `end()` saves nothing unless at least one optimizer step ran. On an exception, `end()` is skipped (no save) and only TensorBoard is stopped.
- `update.sh`/`install.sh` run `pip --upgrade-strategy eager -r requirements-rocm.txt`, which **reinstalls the pinned torch** (`2.13.0+rocm7.2` today) over any manually installed build.
- ROCm bitsandbytes (`libbitsandbytes_rocm72.so`) links hipBLAS/hipSPARSE/hipBLASLt from `/opt/rocm/lib` (a system ROCm 7.2 install) and runs `rocminfo`; without them, import logs a load error and only bnb features fail. bnb picks `libbitsandbytes_rocm<major><minor>` from `torch.version.hip`, so it must match the torch ROCm build.

**Attention**
- `attention_mechanism`: `SDP` = diffusers "native" (torch SDPA picks its own kernel); `FLASH` = diffusers "flash" (the flash-attn package, **CUDA-oriented**); `CUDNN` = **CUDA-only** (not built into ROCm PyTorch; `GenericTrainer.start` stops on it under ROCm); `FLEX` = torch FlexAttention. Chroma always passes a text attention mask when captions are padded (`BaseChromaSetup.predict`); `ChromaModel.encode_text` cuts the text to the batch's longest unmasked length, rounded up to a multiple of 16 when a mask is needed.
- Measured on the 7900 XTX (`rocm_check --full`): SDPA picks FLASH without a mask and EFFICIENT with bool or same-dtype masks (fp32 → EFFICIENT always); a float32 mask with bf16 q/k/v → MATH (up to 30x the memory). All backends match the fp64 reference, including fully masked rows. bf16 linears reach 80-100 TFLOPS; TunableOp made two of three GEMM shapes 1.2-1.3x faster, rocBLAS vs hipBLASLt and `MIOPEN_FIND_MODE=FAST` changed nothing; int W8A8 fwd+bwd ran at 0.77x of bf16.
- **AOTriton bug on gfx11 above head size 256** (measured on the 7900 XTX): flash and memory-efficient SDPA at head size 512 give wrong outputs (rel err 1-17) and NaN gradients in fp32/bf16/fp16; the math kernel is right. PyTorch's `aotriton_max_hdim()` limits gfx11 to 256 only for AOTriton 0.11, and torch 2.13 ships 0.12b. Every VAE's mid-block attention is one 512-wide head, so VAE encodes/decodes (cached latents, sample images) were off (the real Flux/Chroma `ae.safetensors` latent by 9%, a random-init VAE's by 25-40%), and nondeterministically so (that was the "random all-NaN latent" too, not thread concurrency). `modules/util/rocm_sdpa_fix.py` wraps `F.scaled_dot_product_attention` (installed by `script_imports()`) and sends head sizes > 256 on ROCm gfx11 to the math kernel, chunked along the query length (~512 MiB score blocks). `cache_settings` records `vae_attention`, so caches made without the fix rebuild once on such GPUs (None elsewhere, as in older records). `rocm_check` tests PyTorch's raw kernels through `original_scaled_dot_product_attention` (a WARN there is this bug) and OneTrainer's path separately. Confirmed on the 7900 XTX with the random-init and the real Flux/Chroma VAE: with the fix every encode (main thread, worker threads, concurrent) matches the CPU (rel err 3e-6 / 1.8e-5); without it the latent is 0.38 / 0.087 off, first wrong layer `mid_block.attentions.0.to_out.0`.
- ROCm SDPA (torch 2.13 = AOTriton 0.12b): flash and memory-efficient attention are on by default for gfx1100/gfx942/gfx950/gfx1151/gfx1201; gfx1101-1103, gfx1150/1152/1153 and gfx1200 are "experimental" and need `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`, else SDPA silently uses the math kernel. On ROCm only, a float mask whose dtype differs from q/k/v also drops to math (a warning, then slow and memory-hungry); today every model passes a bool mask or one in the train dtype.

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
  - Note: the repo's `requirements-rocm.txt` pins `torch==2.13.0+rocm7.2` (+ bitsandbytes 0.49.2), and `update.sh` will reinstall them (see Gotchas).
- Never suggest CUDA-only fixes (bitsandbytes CUDA paths, xformers, CUDA flash-attn) without flagging them as CUDA-only and offering a ROCm alternative.
- This build supports the `*_ADV` optimizers (ADOPT_ADV, PRODIGY_ADV, ADAMW_ADV). **ADOPT_ADV is the current best performer.** Prodigy-plus-schedule-free needs a CONSTANT scheduler (the code also forces it).
- DoRA stacks with LoRA and LoKr here, but not with LoHa or OFT v2. With LoKr, `lokr_decompose_factor=-1` falls back to the full matrix at dim 32 on 3072-dim weights, so **use factor 16**.
- Main workload: **Chroma1-HD** (`CHROMA_1`) at 512 px, masked training, EMA on, no LR warmup (set `learning_rate_warmup_steps: 0`, since the default is 200).
