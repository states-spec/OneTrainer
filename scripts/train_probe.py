from util.import_util import script_imports

script_imports()

import argparse
import os
from types import SimpleNamespace

from modules.util.config.TrainConfig import TrainConfig
from modules.util.optimizer_util import default_optimizer_config
from modules.util.time_util import get_string_timestamp
from modules.util.train_probe import run_probe

from train import load_config


def main():
    parser = argparse.ArgumentParser(
        description="Train a config for a few optimizer steps and report, per step, the data, the gradients and "
                    "parameter movement per adapter part, Prodigy's D statistics, weight changes outside the "
                    "optimizer and a gradient check. Saves nothing.")
    parser.add_argument("--config-path", required=True, help="the training config (as for train)")
    parser.add_argument("--preset-path", help="a built-in preset applied first (as for train)")
    parser.add_argument("--config-value", action="append", dest="config_values", default=[],
                        help="KEY=VALUE overrides (as for train)")
    parser.add_argument("--steps", type=int, default=40, help="optimizer steps to run (default 40)")
    parser.add_argument("--check-every", type=int, default=5,
                        help="gradient check on the first 3 steps and every N-th (default 5, 0 = off); each one "
                             "costs one more forward+backward and one forward")
    parser.add_argument("--out-dir", help="where the report goes (default diagnostics/train_probe/<timestamp>)")
    args = parser.parse_args()
    if not os.path.isfile(args.config_path):
        parser.error(f"--config-path: no file {os.path.abspath(args.config_path)}")

    load_args = SimpleNamespace(preset_path=args.preset_path, config_path=args.config_path,
                                config_values=args.config_values)
    # as scripts/train.py: on top of the optimizer's UI defaults
    optimizer = load_config(load_args, TrainConfig.default_values()).optimizer.optimizer
    config = TrainConfig.default_values()
    config.optimizer = default_optimizer_config(optimizer)
    config = load_config(load_args, config)

    run_probe(config, args.steps, args.check_every,
              args.out_dir or os.path.join("diagnostics", "train_probe", get_string_timestamp()))


if __name__ == '__main__':
    main()
