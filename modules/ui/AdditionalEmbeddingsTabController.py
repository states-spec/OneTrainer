
from modules.util.config.TrainConfig import TrainConfig, TrainEmbeddingConfig
from modules.util.enum.TrainingMethod import TrainingMethod


class AdditionalEmbeddingsTabController:
    def __init__(self, config: TrainConfig):
        self.train_config = config

    def is_supported(self) -> bool:
        # additional embeddings only train on models that support embedding training
        return TrainingMethod.EMBEDDING in self.train_config.model_type.supported_training_methods()

    def create_new_element(self) -> TrainEmbeddingConfig:
        return TrainEmbeddingConfig.default_values()

    def randomize_uuid(self, embedding_config: TrainEmbeddingConfig) -> TrainEmbeddingConfig:
        embedding_config.uuid = TrainEmbeddingConfig.default_values().uuid
        return embedding_config
