from abc import ABCMeta, abstractmethod

from modules.model.BaseModel import BaseModel
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType

import torch


class BaseModelSaver(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def save(
            self,
            model: BaseModel,
            model_type: ModelType,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        pass

    def check_can_save(self, model: BaseModel, output_model_format: ModelFormat):  # noqa: B027
        # raises if save() would refuse this model in this format; the trainer calls it before training starts.
        # Savers without such refusals keep this default.
        pass
