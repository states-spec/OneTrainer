
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import PeftType
from modules.util.enum.TrainingMethod import TrainingMethod


class LoraTabController:
    def __init__(self, config: TrainConfig):
        self.train_config = config

    def is_active(self) -> bool:
        # the LoRA tab is shown for every training method, but its settings are only used for LoRA training
        return self.train_config.training_method == TrainingMethod.LORA

    def is_supported(self) -> bool:
        return TrainingMethod.LORA in self.train_config.model_type.supported_training_methods()

    def activate(self, ui_state):
        # write the config first: the var's widget callbacks (the top bar's training method dropdown, which rebuilds
        # this tab) run before the trace that copies the var into the config
        self.train_config.training_method = TrainingMethod.LORA
        ui_state.get_var("training_method").set(str(TrainingMethod.LORA))

    def get_peft_types(self) -> list[tuple[str, PeftType]]:
        return [
            ("LoRA", PeftType.LORA),
            ("LoHa", PeftType.LOHA),
            ("OFT v2", PeftType.OFT_2),
            ("LoKr", PeftType.LOKR),
        ]

    def get_lora_weight_dtypes(self) -> list[tuple[str, DataType]]:
        return [
            ("float32", DataType.FLOAT_32),
            ("bfloat16", DataType.BFLOAT_16),
        ]
