import contextlib
import os
from abc import ABCMeta

from modules.model.BaseModel import BaseModel
from modules.util.ModelNames import EmbeddingName, ModelNames

import torch
from torch import Tensor

from safetensors import safe_open
from safetensors.torch import load_file


class EmbeddingLoaderMixin(metaclass=ABCMeta):
    # keys of a LoRA file's bundled embedding (bundle_emb.<placeholder>.<key>, written by the LoRA savers) that this
    # model's embedding files name differently
    _bundle_key_names: dict[str, str] = {}

    def __init__(self):
        super().__init__()

    def __load_embedding(
            self,
            embedding_name: str,
    ) -> dict[str, Tensor] | None:
        if embedding_name == "":
            return None

        with contextlib.suppress(Exception):
            return torch.load(embedding_name, weights_only=True)

        with contextlib.suppress(Exception):
            return load_file(embedding_name)

        raise Exception(f"could not load embedding: {embedding_name}")

    def __load_internal(
            self,
            directory: str,
            embedding_name: EmbeddingName,
    ) -> dict[str, Tensor] | None:
        if os.path.exists(os.path.join(directory, "meta.json")):
            safetensors_embedding_name = os.path.join(
                directory,
                "embeddings",
                f"{embedding_name.uuid}.safetensors",
            )

            if os.path.exists(safetensors_embedding_name):
                return self.__load_embedding(safetensors_embedding_name)
            else:
                return self.__load_embedding(embedding_name.model_name)
        else:
            raise Exception("not an internal model")

    def _load(
            self,
            model: BaseModel,
            directory: str,
            model_names: ModelNames,
    ):
        for embedding_name in model_names.all_embedding():
            try:
                model.embedding_state_dicts[embedding_name.uuid] = \
                    self.__load_internal(directory, embedding_name)
            except Exception as e1:  # noqa: PERF203
                try:
                    model.embedding_state_dicts[embedding_name.uuid] = \
                        self.__load_embedding(embedding_name.model_name)
                except Exception as e2:
                    e2.__cause__ = e1
                    raise Exception(f"could not load embedding: {embedding_name}") from e2

            if model.embedding_state_dicts.get(embedding_name.uuid) is None and embedding_name.model_name == "":
                # no file of its own: continuing from a final LoRA, the embedding is in that file's bundle
                # (bundle_additional_embeddings); before, it started again from its initial text
                bundled = self.__load_bundled(directory, embedding_name.placeholder)
                if bundled is not None:
                    model.embedding_state_dicts[embedding_name.uuid] = bundled
                    print(f"Loaded the embedding '{embedding_name.placeholder}' from the LoRA file '{directory}'")

    def __load_bundled(
            self,
            lora_file: str,
            placeholder: str,
    ) -> dict[str, Tensor] | None:
        if not placeholder or not os.path.isfile(lora_file) or not lora_file.endswith(".safetensors"):
            return None
        prefix = f"bundle_emb.{placeholder}."
        with safe_open(lora_file, framework="pt") as f:
            state = {self._bundle_key_names.get(key[len(prefix):], key[len(prefix):]): f.get_tensor(key)
                     for key in f.keys() if key.startswith(prefix)}  # noqa: SIM118 -- safe_open handle, not a dict
        return state or None
