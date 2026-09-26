from util.import_util import script_imports

script_imports()

import json

from modules.util import create
from modules.util.args.TrainArgs import TrainArgs
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.SecretsConfig import SecretsConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.optimizer_util import default_optimizer_config


def load_config(args: TrainArgs, train_config: TrainConfig) -> TrainConfig:
    if args.preset_path is not None:
        with open(args.preset_path, "r") as f:
            train_config.from_dict(json.load(f), migrate=False)

    with open(args.config_path, "r") as f:
        train_config.from_dict(json.load(f), migrate=args.preset_path is None)

    for config_value in args.config_values or []:
        key, _, value = config_value.partition("=")
        *parent_keys, leaf_key = key.split(".")
        target = train_config
        for parent_key in parent_keys:
            target = getattr(target, parent_key)
        if target.nullables[leaf_key] and value in ("None", "null"):
            value = None
        elif target.types[leaf_key] is bool:
            value = value.lower() in ("true", "1", "yes")
        target.from_dict({leaf_key: value}, migrate=False)

    return train_config


def main():
    args = TrainArgs.parse_args()
    callbacks = TrainCallbacks()
    commands = TrainCommands()

    train_config = load_config(args, TrainConfig.default_values())

    # The UI starts every optimizer from OPTIMIZER_DEFAULT_PARAMETERS. Load again on top of those defaults, so
    # optimizer settings missing from the config files get the UI's values instead of create_optimizer's
    # fallbacks (e.g. beta1=0 for ADOPT_ADV). Settings present in the files still override the defaults.
    optimizer = train_config.optimizer.optimizer
    train_config = TrainConfig.default_values()
    train_config.optimizer = default_optimizer_config(optimizer)
    train_config = load_config(args, train_config)

    try:
        with open("secrets.json" if args.secrets_path is None else args.secrets_path, "r") as f:
            secrets_dict=json.load(f)
            train_config.secrets = SecretsConfig.default_values().from_dict(secrets_dict)
    except FileNotFoundError:
        if args.secrets_path is not None:
            raise

    trainer = create.create_trainer(train_config, callbacks, commands)

    canceled = False
    try:
        trainer.start()
        trainer.train()
    except KeyboardInterrupt:
        canceled = True
    except Exception:
        # end() is skipped on errors, so stop tensorboard here. Otherwise the orphaned subprocess keeps the
        # process (and a cloud orchestrator's pipe) alive after training has already crashed.
        trainer._stop_tensorboard()
        raise

    if not canceled or train_config.backup_before_save:
        trainer.end()
    else:
        trainer._stop_tensorboard()


if __name__ == '__main__':
    main()
