import json
import os
import re

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.path_util import write_json_atomic

CACHE_SETTINGS_FILE = "cache_settings.json"

_TEXT_ENCODER_SETTING = re.compile(r"text_encoder(_\d)?_(layer_skip|sequence_length)")


def cache_settings(config: TrainConfig) -> dict:
    # The settings that decide what the cached latents and text encodings contain. The cache's own keys only cover
    # the concepts (path, seed, image and text settings), so without this a changed model, resolution or dtype
    # would reuse data cached for the old settings.
    parts = {
        name: {
            "model_name": part.model_name,
            "weight_dtype": str(part.weight_dtype),
            "include": part.include,
            "attention_mask": part.attention_mask,
        }
        for name, part in zip(config.model_type.model_parts(), config.model_part_configs(), strict=True)
    }
    # embeddings change the prompts' tokens; the trained one only with embedding training (the uuid names the
    # embedding, it doesn't change what is cached, and a config without one gets a new one every run)
    trained = [config.embedding] if config.training_method == TrainingMethod.EMBEDDING else []
    embeddings = [
        {
            "model_name": embedding.model_name,
            "placeholder": embedding.placeholder,
            "token_count": embedding.token_count,
            "initial_embedding_text": embedding.initial_embedding_text,
            "is_output_embedding": embedding.is_output_embedding,
        }
        for embedding in [*trained, *config.additional_embeddings]
    ]
    return {
        "model_type": str(config.model_type),
        "training_method": str(config.training_method),
        "base_model_name": config.base_model_name,
        "parts": parts,
        "train_dtype": str(config.train_dtype),
        "fallback_train_dtype": str(config.fallback_train_dtype),
        "resolution": config.resolution,
        "frames": config.frames,
        "aspect_ratio_bucketing": config.aspect_ratio_bucketing,
        "masked_training": config.masked_training,
        "custom_conditioning_image": config.custom_conditioning_image,
        "text_encoders": {name: getattr(config, name) for name in sorted(config.types)
                          if _TEXT_ENCODER_SETTING.fullmatch(name)},
        "train_text_encoder_or_embedding": config.train_text_encoder_or_embedding(),
        "embeddings": embeddings,
    }


def changed_cache_settings(config: TrainConfig) -> list[str] | None:
    # the settings that differ from the ones the cache in cache_dir was made with, None if there is no record
    try:
        with open(os.path.join(config.cache_dir, CACHE_SETTINGS_FILE), "r") as f:
            cached = json.load(f)
    except (OSError, ValueError):
        return None
    current = json.loads(json.dumps(cache_settings(config)))
    return sorted(key for key in set(cached) | set(current) if cached.get(key) != current.get(key))


def save_cache_settings(config: TrainConfig):
    os.makedirs(config.cache_dir, exist_ok=True)
    write_json_atomic(os.path.join(config.cache_dir, CACHE_SETTINGS_FILE), cache_settings(config))
