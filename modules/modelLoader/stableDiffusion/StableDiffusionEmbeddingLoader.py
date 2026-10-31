from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelLoader.mixin.EmbeddingLoaderMixin import EmbeddingLoaderMixin
from modules.util.ModelNames import ModelNames


class StableDiffusionEmbeddingLoader(
    EmbeddingLoaderMixin
):
    # the embedding files use the A1111 names, the LoRA bundle the text encoder's
    _bundle_key_names = {"clip_l": "emp_params", "clip_l_out": "emp_params_out"}

    def __init__(self):
        super().__init__()

    def load(
            self,
            model: StableDiffusionModel,
            directory: str,
            model_names: ModelNames,
    ):
        self._load(model, directory, model_names)
