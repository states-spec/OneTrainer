from util.import_util import script_imports

script_imports()

import argparse
import os

from modules.util import path_util
from modules.util.rocm_check.compare import compare


def main():
    parser = argparse.ArgumentParser(
        description="Put rocm_check reports (rocm_check_report.json) side by side, e.g. one from each PyTorch/ROCm "
                    "build: the status of every check, the problems, and every speed number with its speed relative "
                    "to the first report (above 1 = faster).")
    parser.add_argument("reports", nargs="+", help="rocm_check_report.json files; the first is the reference")
    parser.add_argument("--names", nargs="+", help="a name per report (default: its PyTorch build, e.g. 2.13.0+rocm7.2)")
    parser.add_argument("--out", help="where to write the comparison too (default "
                                      "diagnostics/rocm_check/rocm_check_compare.txt)")
    args = parser.parse_args()
    if args.names and len(args.names) != len(args.reports):
        parser.error(f"{len(args.reports)} reports but {len(args.names)} names")
    for path in args.reports:
        if not os.path.isfile(path):
            parser.error(f"no file {os.path.abspath(path)}")
    text = compare(args.reports, args.names)
    print(text)
    out = args.out or path_util.diagnostics_path("rocm_check", "rocm_check_compare.txt")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
