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
    GGUFQuantizationConfig,
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
    ):
        if os.path.isfile(os.path.join(base_model_name, "meta.json")):
            self.__load_diffusers(
                model, model_type, weight_dtypes, base_model_name, transformer_model_name, vae_model_name, quantization,
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
    ):
        tokenizer = Qwen2Tokenizer.from_pretrained(
            base_model_name,
            subfolder="tokenizer",
        )

        t5_tokenizer = T5TokenizerFast.from_pretrained(
            base_model_name,
            subfolder="t5_tokenizer",
        )

        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )

        text_encoder = self._load_transformers_sub_module(
            Qwen3Model,
            weight_dtypes.text_encoder,
            weight_dtypes.fallback_train_dtype,
            base_model_name,
            "text_encoder",
        )

        # conditioner is always bfloat16 — small adapter, no user dtype control
        text_conditioner = AnimaTextConditioner.from_pretrained(
            base_model_name,
            subfolder="text_conditioner",
            torch_dtype=torch.bfloat16,
        )

        if vae_model_name: #TODO simplify
            vae = self._load_diffusers_sub_module(
                AutoencoderKLQwenImage,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                vae_model_name,
            )
        else:
            vae = self._load_diffusers_sub_module(
                AutoencoderKLQwenImage,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                base_model_name,
                "vae",
            )

        if transformer_model_name:
            config_overrides = transformer_config_overrides(
                transformer_model_name,
                CosmosTransformer3DModel.load_config(base_model_name, subfolder="transformer"),
            )
            transformer = CosmosTransformer3DModel.from_single_file(
                transformer_model_name,
                config=base_model_name,
                subfolder="transformer",
                #avoid loading the transformer in float32:
                torch_dtype=torch.bfloat16 if weight_dtypes.transformer.torch_dtype() is None else weight_dtypes.transformer.torch_dtype(),
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16) if weight_dtypes.transformer.is_gguf() else None,
                **config_overrides,
            )
            transformer = self._convert_diffusers_sub_module_to_dtype(
                transformer, weight_dtypes.transformer, weight_dtypes.train_dtype, quantization,
            )
        else:
            transformer = self._load_diffusers_sub_module(
                CosmosTransformer3DModel,
                weight_dtypes.transformer,
                weight_dtypes.train_dtype,
                base_model_name,
                "transformer",
                quantization,
            )

        model.model_type = model_type
        model.tokenizer = tokenizer
        model.t5_tokenizer = t5_tokenizer
        model.noise_scheduler = noise_scheduler
        model.text_encoder = text_encoder
        model.text_conditioner = text_conditioner
        model.vae = vae
        model.transformer = transformer

    def load( #TODO share code between models
            self,
            model: AnimaModel,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ):
        stacktraces = []

        try:
            self.__load_internal(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model, model_names.vae_model, quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_diffusers(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model, model_names.vae_model, quantization,
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
