
from modules.ui.MuonAdamWindowController import MUON_AUX_ADAM_DEFAULTS
from modules.util.config.TrainConfig import TrainConfig, TrainOptimizerConfig
from modules.util.enum.Optimizer import Optimizer
from modules.util.optimizer_util import (
    OPTIMIZER_DEFAULT_PARAMETERS,
    change_optimizer,
    load_optimizer_defaults,
    update_optimizer_config,
)


class OptimizerParamsWindowController:
    def __init__(self, config: TrainConfig):
        self.config = config
        self._loading = False

    def restore_optimizer_config(self, ui_state):
        optimizer_config = change_optimizer(self.config)
        self._load_into_ui(ui_state, optimizer_config)

    def load_defaults(self, ui_state):
        optimizer_config = load_optimizer_defaults(self.config)
        self._load_into_ui(ui_state, optimizer_config)

    def _load_into_ui(self, ui_state, optimizer_config: TrainOptimizerConfig):
        # the settings are written one at a time, and every write runs the widgets' commands, which call on_close:
        # saving then would store a mix of the old and the new optimizer's settings as the new one's defaults
        self._loading = True
        try:
            ui_state.get_var("optimizer").update(optimizer_config)
        finally:
            self._loading = False

    def on_close(self):
        if self._loading:
            return
        update_optimizer_config(self.config)

    def prepare_muon_adam_config(self) -> tuple['TrainOptimizerConfig', Optimizer]:
        current_optimizer = self.config.optimizer.optimizer
        adam_config = TrainOptimizerConfig.default_values()
        current_state = self.config.optimizer.muon_adam_config

        if current_optimizer == Optimizer.MUON:
            defaults = MUON_AUX_ADAM_DEFAULTS
        else:
            defaults = OPTIMIZER_DEFAULT_PARAMETERS[Optimizer.ADAMW_ADV]

        if not current_state:
            adam_config.from_dict(defaults)
            if current_optimizer != Optimizer.MUON:
                adam_config.optimizer = Optimizer.ADAMW_ADV
        elif isinstance(current_state, dict):
            adam_config.from_dict(current_state)
        else:
            # Should not happen if TrainConfig defines it as dict, but for safety
            adam_config = current_state

        return adam_config, current_optimizer

    def save_muon_adam_config(self, adam_config: 'TrainOptimizerConfig'):
        self.config.optimizer.muon_adam_config = adam_config.to_dict()
