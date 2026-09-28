import json
import threading
from abc import ABCMeta

from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ConceptType import ConceptType
from modules.util.TrainProgress import TrainProgress

from mgds.MGDS import MGDS
from mgds.PipelineModule import PipelineModule, PipelineState
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch


def _flatten_modules(definition) -> list[PipelineModule]:
    if isinstance(definition, list):
        return [module for item in definition for module in _flatten_modules(item)]
    return [definition] if isinstance(definition, PipelineModule) else []


def _run_models_one_at_a_time(definition: list):
    # Caching runs the pipeline for "dataloader_threads" samples at once. Loading and cropping images in parallel is
    # fine, but encoding them in parallel runs several forward passes of one model (VAE, text encoder, ...) on the GPU
    # at the same time; on ROCm that gave random all-NaN latents. The modules that hold a model take a shared lock
    # around their own work. Their inputs are fetched before taking it, so the loading stays parallel.
    lock = threading.RLock()  # reentrant: a locked module may pull an uncached input from another one
    for module in _flatten_modules(definition):
        if not isinstance(module, RandomAccessPipelineModule) \
                or not any(isinstance(value, torch.nn.Module) for value in vars(module).values()):
            continue

        def get_item(variation: int, index: int, requested_name: str = None,
                     module=module, unlocked_get_item=module.get_item):
            for name in module.get_inputs():
                # loads the input into the previous modules' (thread-local) item cache, outside the lock
                module._get_previous_item(variation, name, index)
            with lock:
                return unlocked_get_item(variation, index, requested_name)

        module.get_item = get_item


class DataLoaderMgdsMixin(metaclass=ABCMeta):

    def _create_mgds(
            self,
            config: TrainConfig,
            definition: list,
            train_progress: TrainProgress,
            is_validation: bool = False,
    ):
        concepts = config.concepts
        if concepts is None:
            with open(config.concept_file_name, 'r') as f:
                concepts = [ConceptConfig.default_values().from_dict(c) for c in json.load(f)]

        # choose all validation concepts, or none of them, depending on is_validation
        concepts = [concept for concept in concepts if (ConceptType(concept.type) == ConceptType.VALIDATION) == is_validation]

        # convert before passing to MGDS
        concepts = [c.to_dict() for c in concepts]

        settings = {
            "target_resolution": config.resolution,
            "target_frames": config.frames,
        }

        _run_models_one_at_a_time(definition)

        # Just defaults for now.
        ds = MGDS(
            torch.device(config.train_device),
            concepts,
            settings,
            definition,
            batch_size=config.batch_size, #local batch size
            state=PipelineState(config.dataloader_threads),
            initial_epoch=train_progress.epoch,
            initial_epoch_sample=train_progress.epoch_sample,
        )

        return ds
