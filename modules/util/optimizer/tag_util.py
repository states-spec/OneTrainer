import math

from modules.module.LoRAModule import LoRAModuleWrapper

import torch


def tag_peft_parameters(model: torch.nn.Module, oft_scaled: bool):
    """
    Marks adapter parameters with the attributes adv_optm reads to treat them by their role: `_is_lora_A` (down
    projections), `_is_lora_B` (up projections), `_is_dora_scale` (DoRA magnitudes, which are vectors stored as 2D
    tensors) and `_is_oft` (OFT blocks). Spectral scaling, OrthoGrad, Kourkoutas-beta's layer grouping, centered weight
    decay and the state factoring use them; without them a DoRA magnitude is handled like a weight matrix.

    Ported from the adv_optm author's OneTrainer PR #1344 (modules/util/optimizer/tag_util.py). The attributes live
    only on the in-memory Parameter objects: they are not part of any state_dict, so saved files are unchanged.
    """
    def apply_tags(name: str, p: torch.nn.Parameter):
        if name.endswith(("lora_down.weight", "lokr_w1_b", "lokr_w2_b")):
            p._is_lora_A = True
        elif name.endswith(("lora_up.weight", "lokr_w1_a", "lokr_w2_a")):
            p._is_lora_B = True
        elif name.endswith("dora_scale"):
            p._is_dora_scale = True
        elif name.endswith("oft_R.weight"):
            p._is_oft = True
            if oft_scaled:
                # the block size b from the number of rotation parameters per block, b * (b - 1) / 2
                b = (1 + math.sqrt(1 + 8 * p.shape[-1])) / 2
                p._oft_scale_factor = 2 * math.sqrt(b - 1)

    for module in vars(model).values():
        if isinstance(module, LoRAModuleWrapper):
            for lora_module in module.lora_modules.values():
                for name, p in lora_module.named_parameters():
                    apply_tags(name, p)
