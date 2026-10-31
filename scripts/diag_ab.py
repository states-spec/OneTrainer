from util.import_util import script_imports

script_imports()

import argparse
import os

from modules.util import path_util
from modules.util.diag_ab import DEFAULT_VARIANTS, parse_variant, run_ab
from modules.util.diag_report import report
from modules.util.time_util import get_string_timestamp


def main():
    parser = argparse.ArgumentParser(
        description="Train one config under several variants, each with its own workspace, cache, output and "
                    "OT_DIAG_LOG, in alternating order, and compare the runs. Without --variant: baseline vs "
                    "OT_ROCM_MEM_POOLS=true.")
    parser.add_argument("--config-path", required=True, help="the training config (as for train)")
    parser.add_argument("--preset-path", help="a built-in preset applied first (as for train)")
    parser.add_argument("--config-value", action="append", dest="config_values", default=[],
                        help="KEY=VALUE for every run, e.g. epochs=1 (as for train)")
    parser.add_argument("--variant", action="append", dest="variants",
                        help="'NAME env:KEY=VALUE config:key=value python:PATH ...', e.g. 'mempools "
                             "env:OT_ROCM_MEM_POOLS=true', 'nostream config:stream_from_disk=false' or 'rocm71 "
                             "python:~/OT_rocm71/venv/bin/python' (this checkout's code with another venv's PyTorch/"
                             "ROCm build). Can be passed multiple times")
    parser.add_argument("--repeats", type=int, default=2,
                        help="runs per variant (default 2: a repeat shows the run-to-run noise)")
    parser.add_argument("--out-dir", help="where the runs go (default diagnostics/diag_ab/<timestamp>)")
    parser.add_argument("--no-weights", action="store_true", help="don't compare differing saved models tensor by tensor")
    args = parser.parse_args()

    for name, path in (("--config-path", args.config_path), ("--preset-path", args.preset_path)):
        if path and not os.path.isfile(path):
            parser.error(f"{name}: no file {os.path.abspath(path)} (configs saved from the UI's presets dropdown are "
                         f"in training_presets/, the UI's Save config dialog starts in training_configs/)")
    variants = [parse_variant(v) for v in (args.variants or DEFAULT_VARIANTS)]
    if len({v.name for v in variants}) != len(variants):
        parser.error("variant names must differ")
    for v in variants:
        if v.python and not os.path.isfile(v.python):
            parser.error(f"variant {v.name}: no interpreter {v.python} (python:PATH takes another environment's "
                         f"python, e.g. a clone's venv/bin/python)")
    out_dir = args.out_dir or path_util.diagnostics_path("diag_ab", get_string_timestamp())
    logs = run_ab(os.path.abspath(args.config_path), variants, args.repeats, out_dir,
                  os.path.abspath(args.preset_path) if args.preset_path else None, args.config_values)
    logs = [log for log in logs if os.path.exists(log)]
    if not logs:
        raise SystemExit("\nno run got far enough to write a diagnostics log; the end of each run's output.log is above")
    text, _ = report(logs, compare_weights=not args.no_weights)
    print()
    print(text)
    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nreport written to {os.path.join(out_dir, 'report.txt')}")


if __name__ == "__main__":
    main()
