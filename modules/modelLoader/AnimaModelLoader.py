import os
import re
import traceback

from modules.model.AnimaModel import AnimaModel
from modules.modelLoader.GenericFineTuneModelLoader import make_fine_tune_model_loader
from modules.modelLoader.GenericLoRAModelLoader import make_lora_model_loader
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

import torch

from diffusers import (
    AnimaTextConditioner,
    AutoencoderKLQwenImage,
    CosmosTransformer3DModel,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import Qwen2Tokenizer, Qwen3Model, T5TokenizerFast

# top-level transformer blocks in the original (net.blocks.N.), prefix-free (blocks.N.) and diffusers
# (transformer_blocks.N.) layouts; nested blocks such as an LLM adapter's (net.llm_adapter.blocks.N.) don't match
_TRANSFORMER_BLOCK_KEY = re.compile(r"^(?:net\.)?(?:transformer_)?blocks\.(\d+)\.")
_FIRST_BLOCK_Q_KEYS = (
    "net.blocks.0.self_attn.q_proj.weight",
    "blocks.0.self_attn.q_proj.weight",
    "transformer_blocks.0.attn1.to_q.weight",
)


def _read_tensor_shapes(path: str) -> dict[str, tuple[int, ...]] | None:
    # reads only the file header; None for anything that is not a local file (e.g. a URL)
    if not os.path.isfile(path):
        return None
    if path.endswith(".gguf"):
        from gguf import GGUFReader
        return {tensor.name: tuple(int(x) for x in tensor.shape) for tensor in GGUFReader(path).tensors}

    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        return {key: tuple(f.get_slice(key).get_shape()) for key in f.keys()}  # noqa: SIM118


def transformer_config_overrides(transformer_model_name: str, config: dict) -> dict:
    """
    Config values for a single-file transformer whose depth differs from the base model's config, e.g. the 40-block
    Anima 2.9B against the 28-block base model. from_single_file loads with strict=False: without the override, blocks
    beyond the config's num_layers are dropped with only a log warning.
    """
    shapes = _read_tensor_shapes(transformer_model_name)
    if not shapes:
        return {}

    indices = {int(m.group(1)) for key in shapes if (m := _TRANSFORMER_BLOCK_KEY.match(key))}
    if not indices:
        return {}
    num_layers = max(indices) + 1
    if indices != set(range(num_layers)):
        raise ValueError(f"The transformer file {transformer_model_name} has gaps in its block numbering "
                         f"({len(indices)} blocks, highest index {num_layers - 1}).")

    hidden_size = config["num_attention_heads"] * config["attention_head_dim"]
    q_shape = next((shapes[key] for key in _FIRST_BLOCK_Q_KEYS if key in shapes), None)
    if q_shape is not None and hidden_size not in q_shape:
        raise ValueError(f"The transformer file {transformer_model_name} has a block width of {max(q_shape)}, but the "
                         f"base model's transformer config has {hidden_size}. Only the number of blocks is detected "
                         f"from the file; use a base model with the same width.")

    if num_layers == config["num_layers"]:
        return {}
    print(f"The transformer file has {num_layers} blocks, the base model's config {config['num_layers']}: "
          f"loading it with {num_layers} blocks.")
    return {"num_layers": num_layers}


class AnimaModelLoader(
    HFModelLoaderMixin,
):
    def __init__(self):
        super().__init__()

    def __load_internal(
            self,
            model: AnimaModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
            stream_from_disk: bool,
    ):
        if os.path.isfile(os.path.join(base_model_name, "meta.json")):
            self.__load_diffusers(
                model, model_type, weight_dtypes, base_model_name, transformer_model_name, vae_model_name, quantization,
                stream_from_disk,
            )
        else:
            raise Exception("not an internal model")

    def __load_diffusers(
            self,
            model: AnimaModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
            stream_from_disk: bool,
    ):
        model.tokenizer = Qwen2Tokenizer.from_pretrained(
            base_model_name,
            subfolder="tokenizer",
        )

        model.t5_tokenizer = T5TokenizerFast.from_pretrained(
            base_model_name,
            subfolder="t5_tokenizer",
        )

        model.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )

        model.text_encoder, model.materialize_fn["text_encoder"] = self._load_text_encoder(
            Qwen3Model,
            weight_dtypes.text_encoder,
            weight_dtypes.fallback_train_dtype,
            base_model_name,
            "text_encoder",
            stream_from_disk=stream_from_disk,
        )

        # conditioner is always bfloat16 — small adapter, no user dtype control
        model.text_conditioner = AnimaTextConditioner.from_pretrained(
            base_model_name,
            subfolder="text_conditioner",
            torch_dtype=torch.bfloat16,
        )

        model.vae = self._load_vae(
            AutoencoderKLQwenImage,
            weight_dtypes.vae,
            weight_dtypes.train_dtype,
            base_model_name,
            vae_model_name,
        )

        # a single file can have more blocks than the base model's config (the 40-block 2.9B on the 28-block base);
        # from_single_file loads with strict=False and would drop them, so the block count is read from the file
        single_file_overrides = transformer_config_overrides(
            transformer_model_name,
            CosmosTransformer3DModel.load_config(base_model_name, subfolder="transformer"),
        ) if transformer_model_name else None
        model.transformer, model.materialize_fn["transformer"] = self._load_transformer(
            CosmosTransformer3DModel,
            weight_dtypes,
            base_model_name,
            transformer_model_name,
            quantization,
            config=base_model_name,
            stream_from_disk=stream_from_disk,
            single_file_overrides=single_file_overrides,
        )

    def load( #TODO share code between models
            self,
            model: AnimaModel,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
            stream_from_disk: bool = False,
    ):
        stacktraces = []

        try:
            self.__load_internal(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model, model_names.vae_model, quantization,
                stream_from_disk,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_diffusers(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model, model_names.vae_model, quantization,
                stream_from_disk,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load model: " + model_names.base_model)


class AnimaLoRALoader(
    LoRALoaderMixin,
):
    def __init__(self):
        super().__init__()

    def load(
            self,
            model: AnimaModel,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)


AnimaLoRAModelLoader = make_lora_model_loader(
    model_spec_map={ModelType.ANIMA: "resources/sd_model_spec/anima-lora.json"},
    model_class=AnimaModel,
    model_loader_class=AnimaModelLoader,
    embedding_loader_class=None,
    lora_loader_class=AnimaLoRALoader,
)

AnimaFineTuneModelLoader = make_fine_tune_model_loader(
    model_spec_map={ModelType.ANIMA: "resources/sd_model_spec/anima.json"},
    model_class=AnimaModel,
    model_loader_class=AnimaModelLoader,
    embedding_loader_class=None,
)
