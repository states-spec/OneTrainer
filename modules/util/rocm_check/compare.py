"""Puts rocm_check reports side by side (scripts/rocm_check_compare.py), e.g. one per PyTorch/ROCm build to choose
between them: which checks pass in each, and every speed number with its speed relative to the first report.

A report's numbers come from one run: the run-time switches' child processes vary between runs of the same setup (see
that check's note), and the first run in a new environment pays for empty kernel caches (MIOpen's find database,
Triton's autotuning) in the first-call and compile numbers. Two reports from the same setup show that noise."""
import json
import math
import os

from modules.util.rocm_check.framework import FAIL, WARN, stack_label

# checks whose speed headline uses only some of their numbers: the attention benchmark also times each backend forced
# on its own (flash, efficient, math), and training uses PyTorch's own choice
_HEADLINE_KEYS = {"attention fwd+bwd per model family and backend": " default ms"}
# no headline: the run-time switches compare settings within one run, and their child processes' times vary between
# runs of the same setup
_NO_HEADLINE = ("runtime switches",)
# numbers that time a kernel search, a compile or a cold cache, not the steady state: shown, not in the headline
_NOT_STEADY = ("first call", "compile seconds")


def _direction(key: str) -> int:
    """-1: a time (lower is faster), 1: a throughput (higher is faster), 0: anything else (memory, errors, versions)"""
    k = key.lower()
    if "tflops" in k or "gb/s" in k:
        return 1
    if k.endswith((" ms", "seconds")) or " ms " in k or k.startswith("ms "):
        return -1
    return 0


def _stack(report: dict) -> dict:
    # the header has it since the report header named the PyTorch build; before, only the environment check did
    if report.get("torch"):
        return {k: report.get(k) for k in ("torch", "hip", "rocm", "cuda")}
    env = next((r for r in report["results"] if r["name"].startswith("PyTorch, GPU, libraries")), None)
    metrics = env["metrics"] if env else {}
    return {"torch": metrics.get("torch"), "hip": metrics.get("hip"), "rocm": metrics.get("rocm")}


def _onetrainer(report: dict) -> str:
    if report.get("onetrainer"):
        return report["onetrainer"]
    env = next((r for r in report["results"] if r["name"].startswith("PyTorch, GPU, libraries")), None)
    line = next((line for line in (env["lines"] if env else []) if line.startswith("OneTrainer ")), "")
    return line.removeprefix("OneTrainer ") or "?"


def _first_problem(result: dict) -> str:
    status = result["status"]
    line = next((line for line in result["lines"] if line.startswith(status + ": ")), "")
    line = line.removeprefix(status + ": ").splitlines()[0] if line else ""
    return line if len(line) <= 160 else line[:157] + "..."


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3g}" if abs(v) < 1000 else f"{v:.0f}"
    return str(v)


def _table(rows: list[list[str]]) -> str:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip() for r in rows)


def _relative_speed(key: str, base, value) -> float | None:
    """how fast value is compared with base: above 1 is faster, below 1 slower"""
    direction = _direction(key)
    if direction == 0 or not isinstance(base, int | float) or not isinstance(value, int | float) \
            or isinstance(base, bool) or base <= 0 or value <= 0:
        return None
    return base / value if direction < 0 else value / base


def _geomean(values: list[float]) -> float | None:
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else None


def labels_for(reports: list[dict], paths: list[str]) -> list[str]:
    """the PyTorch build of each report (torch 2.13.0+rocm7.2 -> '2.13.0+rocm7.2'), numbered when two share one"""
    labels = []
    for report, path in zip(reports, paths, strict=True):
        label = _stack(report).get("torch") or os.path.splitext(os.path.basename(path))[0]
        n, unique = 2, label
        while unique in labels:
            unique, n = f"{label} #{n}", n + 1
        labels.append(unique)
    return labels


def compare(paths: list[str], names: list[str] | None = None) -> str:
    reports = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            reports.append(json.load(f))
    labels = names or labels_for(reports, paths)
    by_name = [{(r["section"], r["name"]): r for r in report["results"]} for report in reports]
    checks = []
    for report in reports:
        for r in report["results"]:
            if (r["section"], r["name"]) not in checks:
                checks.append((r["section"], r["name"]))

    out = ["reports:"]
    for label, report, path in zip(labels, reports, paths, strict=True):
        counts = {}
        for r in report["results"]:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        out.append(f"  {label}: {stack_label(_stack(report)) or 'PyTorch build not recorded'}; OneTrainer "
                   f"{_onetrainer(report)}; {report.get('mode')} mode, started {report.get('started')}; "
                   + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + f"  ({path})")
        if report.get("running_when_written"):
            out.append(f"    !! stopped or still running in: {report['running_when_written']}")

    out += ["", "status per check (* = differs between the reports; - = not in that report):"]
    rows = [["", *labels, "check"]]
    for key in checks:
        statuses = [found["status"] if (found := b.get(key)) else "-" for b in by_name]
        ran = {s for s in statuses if s != "-"}
        rows.append(["*" if len(ran) > 1 else "", *statuses, f"{key[0]} / {key[1]}"])
    out.append(_table(rows))

    problems = []
    for label, b in zip(labels, by_name, strict=True):
        for key in checks:
            r = b.get(key)
            if r and r["status"] in (FAIL, WARN):
                problems.append(f"  {label}: [{r['status']}] {key[1]}: {_first_problem(r)}")
    if problems:
        out += ["", "problems:", *problems]

    # speed: every timed or throughput number, with each report's speed relative to the first (above 1 = faster)
    speed_rows, details = [], []
    for key in checks:
        metrics = [b[key]["metrics"] if key in b else {} for b in by_name]
        names_in_order = []
        for m in metrics:
            names_in_order += [k for k in m if k not in names_in_order and _direction(k) != 0]
        if not names_in_order:
            continue
        headline_filter = _HEADLINE_KEYS.get(key[1])
        in_headline = not key[1].startswith(_NO_HEADLINE)
        ratios = [[] for _ in reports]
        rows = [["", *labels]]
        for name in names_in_order:
            cells = []
            for i, m in enumerate(metrics):
                value = m.get(name)
                rel = _relative_speed(name, metrics[0].get(name), value) if i > 0 else None
                if rel is not None and in_headline and not any(s in name for s in _NOT_STEADY) \
                        and (headline_filter is None or name.endswith(headline_filter)):
                    ratios[i].append(rel)
                cells.append(_fmt(value) + (f" ({rel:.2f}x)" if rel is not None else ""))
            rows.append([name, *cells])
        headline = [_geomean(r) for r in ratios]
        if any(h is not None for h in headline[1:]):
            speed_rows.append([key[1], "1", *[f"{h:.2f}x" if h is not None else "-" for h in headline[1:]]])
        details += ["", f"{key[1]}:", _table(rows)]

    if len(reports) > 1 and speed_rows:
        out += ["", f"speed relative to {labels[0]} (geometric mean of each check's numbers; above 1 = faster):",
                _table([["check", *labels], *speed_rows])]
        attention = _HEADLINE_KEYS["attention fwd+bwd per model family and backend"].strip()
        out.append(f"(attention: PyTorch's own backend choice only, '{attention}'; without first calls and compile "
                   f"times; the run-time switches are left out, their child processes vary between runs. Numbers in "
                   f"parentheses below: speed relative to {labels[0]})")
    if details:
        out += ["", "numbers per check:", *details]
    return "\n".join(out) + "\n"
