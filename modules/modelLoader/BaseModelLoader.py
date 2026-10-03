from abc import ABCMeta, abstractmethod

from modules.model.BaseModel import BaseModel
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes


class BaseModelLoader(metaclass=ABCMeta):

    @abstractmethod
    def load(
            self,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ) -> BaseModel | None:
        pass
