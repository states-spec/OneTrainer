from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.TrainingMethod import TrainingMethod

_NAMES = {
    TrainingMethod.LORA: "LoRA",
    TrainingMethod.EMBEDDING: "Embedding",
}


def activate_training_method(train_config: TrainConfig, ui_state, method: TrainingMethod):
    # write the config first: the var's widget callbacks (the top bar's training method dropdown, which rebuilds the
    # method tabs) run before the trace that copies the var into the config
    train_config.training_method = method
    ui_state.get_var("training_method").set(str(method))


def build_training_method_notice(
        components,
        frame,
        row: int,
        column: int,
        train_config: TrainConfig,
        ui_state,
        method: TrainingMethod,
        columnspan: int = 1,
) -> bool:
    # For a tab whose settings only apply to one training method. The tab stays visible for every method; while
    # another one is selected, say why its settings are unused and offer to switch. Returns whether they are used.
    if train_config.training_method == method:
        return True

    name = _NAMES[method]
    notice = components.inline_frame(frame, row, column, columnspan)
    if method in train_config.model_type.supported_training_methods():
        components.label(notice, 0, 0, f"Not used: the training method (top right) is not {name}",
                         tooltip=f"These settings only apply to {name} training. They are kept, and used again once {name} is selected.")
        components.button(notice, 0, 1, f"Switch to {name}", lambda: activate_training_method(train_config, ui_state, method),
                          tooltip=f"Set the training method to {name}", sticky="nw")
    else:
        components.label(notice, 0, 0, f"Not used: this model type does not support {name} training")
    return False
