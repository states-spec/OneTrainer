from modules.util.enum.Optimizer import Optimizer
from modules.util.optimizer_util import adv_state_precisions


class BaseMuonAdamWindowView:
    def __init__(self, components):
        self.components = components

    def build_content(self, master, controller, ui_state):
        # This is a large map, copied from OptimizerParamsWindow for simplicity.
        # @formatter:off
        KEY_DETAIL_MAP = {
            'alpha': {'title': 'Alpha', 'tooltip': 'Smoothing parameter for RMSprop and others.', 'type': 'float'},
            'beta1': {'title': 'Beta1', 'tooltip': 'Momentum term.', 'type': 'float'},
            'beta2': {'title': 'Beta2', 'tooltip': 'Coefficients for computing running averages of gradient.', 'type': 'float'},
            'eps': {'title': 'EPS', 'tooltip': 'A small value to prevent division by zero.', 'type': 'float'},
            'stochastic_rounding': {'title': 'Stochastic Rounding', 'tooltip': 'Stochastic rounding for weight updates. Improves quality when using bfloat16 weights.', 'type': 'bool'},
            'use_bias_correction': {'title': 'Bias Correction', 'tooltip': 'Turn on Adam\'s bias correction.', 'type': 'bool'},
            'weight_decay': {'title': 'Weight Decay', 'tooltip': 'Regularization to prevent overfitting.', 'type': 'float'},
            'use_orthograd': {'title': 'OrthoGrad', 'tooltip': 'Experimental. OrthoGrad (arXiv 2501.04697): update with only the gradient component orthogonal to the weights. Can reduce overfitting.', 'type': 'bool'},
            'nnmf_factor': {'title': 'Factored Optimizer', 'tooltip': 'Enables a memory-efficient mode by applying fast low-rank factorization to the optimizers states. It combines factorization for magnitudes with 1-bit compression for signs, drastically reducing VRAM usage and allowing for larger models or batch sizes. This is an approximation which may slightly alter training dynamics.', 'type': 'bool'},
            'nesterov': {'title': 'Nesterov', 'tooltip': 'Whether to use Nesterov momentum. Its mixing is set by Nesterov Coefficient.', 'type': 'bool'},
            'nesterov_coef': {'title': 'Nesterov Coefficient', 'tooltip': 'Mixing coefficient of the Nesterov update: coef * momentum + (1 - coef) * gradient. Empty uses Beta1 (or the momentum value). Lower values weight the current gradient more; this replaces the former Simplified AdEMAMix "Grad α". Only used when Nesterov is on and Beta1/momentum is above 0.', 'type': 'float'},
            'state_precision': {'title': 'State Precision', 'tooltip': 'Storage format of the optimizer state. auto: the parameter dtype (the former behavior). fp32: full precision. factored: rank-1 factored low-memory state (the former Factored Optimizer option). bf16_sr / int8_sr: bfloat16 / 8-bit state with stochastic rounding. fp16: half precision (ADOPT_ADV only). Only the modes that work with the selected optimizer are listed.', 'type': 'choice', 'values': adv_state_precisions},
            'factored_2nd': {'title': 'Factored 2nd Moment', 'tooltip': 'Factorizes only the second moment (the variance estimate) to save memory, while the first moment keeps the State Precision format. Works together with any State Precision.', 'type': 'bool'},
            'fisher_wd': {'title': 'Fisher Weight Decay', 'tooltip': 'Scales the weight decay by the second-moment (Fisher) estimate, as in the FAdam paper, instead of applying it uniformly.', 'type': 'bool'},
            'centered_wd': {'title': 'Centered Weight Decay', 'tooltip': 'Decays the weights toward their values at the start of training (the anchor) instead of toward zero. Can be combined with the normal weight decay. 0 or empty disables it. Stores a copy of the trained weights, see Anchor Precision.', 'type': 'float'},
            'centered_wd_mode': {'title': 'Anchor Precision', 'tooltip': 'Storage format of the Centered Weight Decay anchor. full: the parameter dtype. float8: float8 e4m3 (default). int8 / int4: block-wise quantized. Only used when Centered Weight Decay is above 0.', 'type': 'choice', 'values': ['full', 'float8', 'int8', 'int4']},
            'spectral_normalization': {'title': 'Spectral Scaling', 'tooltip': 'Scales each update by its spectral norm (estimated by power iteration), which makes the update size independent of the layer width or LoRA rank.', 'type': 'bool'},
            'orthogonal_gradient': {'title': 'OrthoGrad', 'tooltip': 'Removes the gradient component parallel to the weight, which reduces overfitting and improves generalization. flattened: the original OrthoGrad over the whole tensor. iterative: a matrix-wise variant (adv_optm 2.5). disabled: off.', 'type': 'choice', 'values': ['disabled', 'flattened', 'iterative']},
            'use_atan2': {'title': 'Atan2 Scaling', 'tooltip': 'A robust replacement for eps, which also incorporates gradient clipping, bounding and stabilizing the optimizer updates.', 'type': 'bool'},
            'kourkoutas_beta': {'title': 'Kourkoutas Beta', 'tooltip': 'Enables a layer-wise dynamic β₂ adaptation. This feature makes the optimizer more responsive to "spiky" gradients by lowering β₂ during periods of high variance, and more stable during calm periods by raising β₂ towards its maximum. It can significantly improve training stability and final loss.', 'type': 'bool'},
        }
        # @formatter:on

        adam_params = controller.get_adam_params_def()

        for index, key in enumerate(adam_params.keys()):
            if key not in KEY_DETAIL_MAP:
                continue

            arg_info = KEY_DETAIL_MAP[key]

            title = arg_info['title']
            tooltip = arg_info['tooltip']
            param_type = arg_info['type']

            row = index // 2
            col = 3 * (index % 2)

            self.components.label(master, row, col, title, tooltip=tooltip)

            if param_type == 'choice':
                values = arg_info['values'](Optimizer.ADAMW_ADV) if callable(arg_info['values']) else arg_info['values']
                self.components.options(master, row, col + 1, values, ui_state, key)
            elif param_type != 'bool':
                self.components.entry(master, row, col + 1, ui_state, key)
            else:
                self.components.switch(master, row, col + 1, ui_state, key)
