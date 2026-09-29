from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch


class CheckFinite(
    PipelineModule,
    RandomAccessPipelineModule,
):
    # Passes its names through unchanged and raises if a floating point tensor among them holds NaN or inf. Placed in
    # front of a disk cache, it stops caching at the sample that produced the bad value (e.g. a VAE encoding that
    # came out NaN) instead of letting training run into it later, where it only shows up as a NaN loss or a crash.
    def __init__(self, names: list[str], path_in_name: str = 'image_path'):
        super().__init__()
        self.names = names
        self.path_in_name = path_in_name

    def length(self) -> int:
        return self._get_previous_length(self.names[0])

    def get_inputs(self) -> list[str]:
        return self.names + [self.path_in_name]

    def get_outputs(self) -> list[str]:
        return self.names

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        value = self._get_previous_item(variation, requested_name, index)

        if isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
            path = self._get_previous_item(variation, self.path_in_name, index)
            nan_count = torch.isnan(value).sum().item()
            inf_count = torch.isinf(value).sum().item()
            raise RuntimeError(
                f'"{requested_name}" of {path} contains {nan_count} NaN and {inf_count} inf values out of '
                f'{value.numel()}, so caching stopped before anything was trained on it. If the file is a normal '
                f'image, the encoder produced the bad values.'
            )

        return {
            requested_name: value,
        }
