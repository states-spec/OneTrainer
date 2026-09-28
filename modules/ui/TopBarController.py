import json
import os
import shutil
import traceback
import webbrowser
from contextlib import suppress

from modules.util import path_util
from modules.util.config.SecretsConfig import SecretsConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.path_util import write_json_atomic


class TopBarController:
    def __init__(self, config: TrainConfig):
        self.train_config = config

    def get_model_types(self) -> list[tuple[str, ModelType]]:
        return [  #TODO simplify
            ("SD1.5", ModelType.STABLE_DIFFUSION_15),
            ("SD1.5 Inpainting", ModelType.STABLE_DIFFUSION_15_INPAINTING),
            ("SD2.0", ModelType.STABLE_DIFFUSION_20),
            ("SD2.0 Inpainting", ModelType.STABLE_DIFFUSION_20_INPAINTING),
            ("SD2.1", ModelType.STABLE_DIFFUSION_21),
            ("SD3", ModelType.STABLE_DIFFUSION_3),
            ("SD3.5", ModelType.STABLE_DIFFUSION_35),
            ("SDXL", ModelType.STABLE_DIFFUSION_XL_10_BASE),
            ("SDXL Inpainting", ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING),
            ("Wuerstchen v2", ModelType.WUERSTCHEN_2),
            ("Stable Cascade", ModelType.STABLE_CASCADE_1),
            ("PixArt Alpha", ModelType.PIXART_ALPHA),
            ("PixArt Sigma", ModelType.PIXART_SIGMA),
            ("Flux Dev.1", ModelType.FLUX_DEV_1),
            ("Flux Fill Dev", ModelType.FLUX_FILL_DEV_1),
            ("Flux 2 [Dev, Klein]", ModelType.FLUX_2),
            ("Sana", ModelType.SANA),
            ("Hunyuan Video", ModelType.HUNYUAN_VIDEO),
            ("HiDream Full", ModelType.HI_DREAM_FULL),
            ("Chroma1", ModelType.CHROMA_1),
            ("QwenImage", ModelType.QWEN),
            ("Anima", ModelType.ANIMA),
            ("Krea 2", ModelType.KREA_2),
            ("Z-Image", ModelType.Z_IMAGE),
            ("Ernie Image", ModelType.ERNIE),
            ("Ideogram 4", ModelType.IDEOGRAM_4),
        ]

    def get_training_methods(self, model_type: ModelType) -> list[tuple[str, TrainingMethod]]:
        labels = {
            TrainingMethod.FINE_TUNE: "Fine Tune",
            TrainingMethod.LORA: "LoRA",
            TrainingMethod.EMBEDDING: "Embedding",
            TrainingMethod.FINE_TUNE_VAE: "Fine Tune VAE",
        }
        return [(labels[m], m) for m in model_type.supported_training_methods()]

    def load_preset_tree(self, dir: str = "training_presets") -> list[tuple[str, str | list]]:
        # mirrors whatever directory structure happens to exist under `dir`; a node is either
        # (display_name, path) for a leaf preset or (display_name, children) for a group.
        # "#" marks a built-in preset (vs. a user file); "#.json" is the last-session state,
        # not a preset. Same convention as load_config_from_file's is_built_in_preset check.
        nodes = []
        if os.path.isdir(dir):
            for entry in sorted(os.scandir(dir), key=lambda e: e.name.lower()):
                if entry.is_dir():
                    children = self.load_preset_tree(entry.path)
                    if children:
                        nodes.append((entry.name, children))
                elif entry.name.startswith("#") and entry.name != "#.json" and entry.name.endswith(".json"):
                    nodes.append((os.path.splitext(entry.name)[0], path_util.canonical_join(dir, entry.name)))
        return nodes

    def preset_base_models(self, dir: str = "training_presets") -> dict[ModelType, list[str]]:
        # the base models the built-in presets use per model type, most used first
        counts: dict[ModelType, dict[str, int]] = {}
        for root, _, files in os.walk(dir):
            for name in files:
                if not (name.startswith("#") and name != "#.json" and name.endswith(".json")):
                    continue
                with suppress(Exception), open(os.path.join(root, name), "r") as f:
                    preset = json.load(f)
                    model_type = ModelType(preset.get("model_type", str(ModelType.STABLE_DIFFUSION_15)))
                    base_model = preset.get("base_model_name")
                    if base_model:
                        type_counts = counts.setdefault(model_type, {})
                        type_counts[base_model] = type_counts.get(base_model, 0) + 1
        return {model_type: sorted(names, key=lambda n: -names[n]) for model_type, names in counts.items()}

    def update_base_model_for(self, model_type: ModelType, ui_state):
        # Switching the model type keeps the base model, so a base model left from another model type would be loaded
        # (and downloaded) as this one. Replace it only while it is another model type's preset default (or empty);
        # a model the user picked stays.
        preset_base_models = self.preset_base_models()
        current = self.train_config.base_model_name
        own_defaults = preset_base_models.get(model_type, [])
        other_defaults = {name for other_type, names in preset_base_models.items() if other_type != model_type
                          for name in names}
        if current in own_defaults or (current and current not in other_defaults):
            return
        new = own_defaults[0] if own_defaults else ""
        if new != current:
            self.train_config.base_model_name = new  # config first: the var's callbacks run before its trace
            ui_state.get_var("base_model_name").set(new)

    def save_to_file(self, name) -> str:
        name = path_util.safe_filename(name)
        path = path_util.canonical_join("training_presets", f"{name}.json")
        write_json_atomic(path, self.train_config.to_settings_dict(secrets=False))
        return path

    def save_config_to_path(self, path: str) -> None:
        write_json_atomic(path, self.train_config.to_settings_dict(secrets=False))

    def load_config_from_file(self, filename: str) -> TrainConfig | None:
        try:
            basename = os.path.basename(filename)
            is_built_in_preset = basename.startswith("#") and basename != "#.json"

            with open(filename, "r") as f:
                loaded_dict = json.load(f)
                default_config = TrainConfig.default_values()
                # built-in configs are always saved in the most recent version, so migration can be skipped
                loaded_config = default_config.from_dict(loaded_dict, migrate=not is_built_in_preset).to_unpacked_config()

            with suppress(FileNotFoundError), open("secrets.json", "r") as f:
                secrets_dict = json.load(f)
                loaded_config.secrets = SecretsConfig.default_values().from_dict(secrets_dict)

            self.train_config.from_dict(loaded_config.to_dict())
            return loaded_config
        except FileNotFoundError:
            return None
        except Exception:
            print(traceback.format_exc())
            if os.path.basename(filename) == "#.json":
                # the UI keeps its defaults and overwrites #.json with them on close, so keep the unreadable state
                with suppress(OSError):
                    shutil.copyfile(filename, filename + ".bak")
                    print(f"Could not restore the last session. A copy of {filename} was saved as {filename}.bak")
            return None

    def save_secrets(self, path) -> str:
        write_json_atomic(path, self.train_config.secrets.to_dict())
        return path

    def open_wiki(self):
        webbrowser.open("https://github.com/Nerogar/OneTrainer/wiki", new=0, autoraise=False)

    def save_default(self):
        self.save_to_file("#")
        self.save_secrets("secrets.json")
