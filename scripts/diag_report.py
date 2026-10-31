from util.import_util import script_imports

script_imports()

import argparse
import json

from modules.util.diag_report import report


def main():
    parser = argparse.ArgumentParser(description="Summarize OT_DIAG_LOG files and compare runs: settings that differ, "
                                                 "GPU memory and fragmentation per phase, step times, losses and the "
                                                 "saved weights. The first log is the reference.")
    parser.add_argument("logs", nargs="+", help="OT_DIAG_LOG .jsonl files")
    parser.add_argument("--json", dest="json_path", help="also write the numbers to this .json file")
    parser.add_argument("--no-weights", action="store_true",
                        help="don't load differing saved models to measure how far apart they are")
    args = parser.parse_args()
    text, data = report(args.logs, compare_weights=not args.no_weights)
    print(text)
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)


if __name__ == "__main__":
    main()
