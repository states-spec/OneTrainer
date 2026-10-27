"""Summarizes OT_DIAG_LOG files (modules/util/run_diagnostics.py) and compares runs: settings that differ, memory and
fragmentation per phase, step times, where the losses part, and whether the saved weights are the same."""
import json
import os
import statistics
from dataclasses import dataclass, field
from typing import Any

GIB = 2 ** 30
# settings that differ between runs by design (each run has its own folders and log) and say nothing about the run
_IGNORED_KEYS = ("workspace_dir", "cache_dir", "final_output_dir", "run_name", "env.OT_DIAG_LOG", "argv",
                 "debug_dir", "quantization.cache_dir")


@dataclass
class Run:
    path: str
    label: str
    variant: str
    header: dict = field(default_factory=dict)
    records: list = field(default_factory=list)

    def of(self, kind: str) -> list[dict]:
        return [r for r in self.records if r["kind"] == kind]


def load(path: str) -> Run:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    break  # a line cut off by a crash ends the usable part
    # a log can hold several runs (the UI appends); the last one is the one to look at
    starts = [i for i, r in enumerate(records) if r["kind"] == "header"]
    if starts:
        records = records[starts[-1]:]
    header = records[0] if records and records[0]["kind"] == "header" else {}
    env = header.get("env", {})
    name = os.path.splitext(os.path.basename(path))[0]
    if name == "diag":
        name = os.path.basename(os.path.dirname(os.path.abspath(path)))
    return Run(path=path, label=name, variant=env.get("OT_DIAG_LABEL", name), header=header, records=records)


def _flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}{k}."))
    elif isinstance(d, list):
        out[prefix[:-1]] = json.dumps(d, sort_keys=True, default=str)
    else:
        out[prefix[:-1]] = d
    return out


def settings(run: Run) -> dict[str, Any]:
    return _flatten({k: v for k, v in run.header.items() if k not in ("kind", "t")})


def _q(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def summarize(run: Run) -> dict[str, Any]:
    steps = run.of("step")
    full = run.of("phase") + run.of("move") + run.of("error")
    errors = run.of("error")
    gpu = any("reserved" in r for r in run.records)
    status = (f"error: {errors[-1]['type']}" + (" (out of memory)" if errors[-1].get("out_of_memory") else "")
              + f": {errors[-1].get('message', '')[:300]}") \
        if errors else ("finished" if run.of("end") or run.of("output") else "incomplete (crash, hang or killed?)")
    # the first two steps compile kernels and fill caches; steps with sampling/backups/part moves in between aren't
    # a step time
    clean = [s["seconds"] for s in steps[2:] if s.get("clean") and s.get("seconds") is not None]
    out: dict[str, Any] = {
        "status": status,
        "steps": len(steps),
        "loss mean": statistics.fmean(s["loss"] for s in steps) if steps else None,
        "step s median": _q(clean, 0.5),
        "step s p90": _q(clean, 0.9),
        "run s": run.records[-1]["t"] if run.records else None,
        "RAM peak GiB": max((r["ram"] for r in run.records if r.get("ram")), default=0) / GIB or None,
    }
    if gpu:
        mem = [r for r in run.records if "reserved" in r]
        held = [(r["reserved"] - r["allocated"]) for r in full if "reserved" in r]
        out.update({
            "peak allocated GiB": max(r["peak_allocated"] for r in mem) / GIB,
            "peak reserved GiB": max(r["peak_reserved"] for r in mem) / GIB,
            "held unused max GiB": max(held, default=0) / GIB,
            "held unused at end GiB": held[-1] / GIB if held else None,
            "free on device min GiB": min(r["device_free"] for r in mem) / GIB,
            "frag max": max((r.get("ext_frag", 0) for r in full), default=0),
            "inactive split max GiB": max(r.get("inactive_split", 0) for r in mem) / GIB,
            "MemPool reserved max GiB": max((r.get("pool_reserved", 0) for r in full), default=0) / GIB,
            "alloc retries": mem[-1].get("alloc_retries"),
            "OOMs": mem[-1].get("ooms"),
            "device mallocs": mem[-1].get("device_allocs"),
            "device frees": mem[-1].get("device_frees"),
            "GPU temp max C": max((r["temperature_c"] for r in run.records if "temperature_c" in r), default=None),
            "GPU clock min MHz": min((r["clock_mhz"] for r in run.records if "clock_mhz" in r), default=None),
        })
    moves = run.of("move")
    out["part moves"] = len(moves)
    out["part move s"] = sum(m["seconds"] for m in moves)
    phases: dict[str, dict[str, float]] = {}
    for p in run.of("phase"):
        entry = phases.setdefault(p["name"], {"count": 0, "seconds": 0.0, "peak reserved GiB": None,
                                             "held unused max GiB": None})
        entry["count"] += 1
        entry["seconds"] += p["seconds"]
        if "peak_reserved" in p:
            entry["peak reserved GiB"] = max(entry["peak reserved GiB"] or 0.0, p["peak_reserved"] / GIB)
            entry["held unused max GiB"] = max(entry["held unused max GiB"] or 0.0,
                                               (p["reserved"] - p["allocated"]) / GIB)
    out["phases"] = phases
    outputs = run.of("output")
    out["output"] = outputs[-1] if outputs else None
    return out


_LEGEND = """reading the GPU numbers:
  peak reserved: the most GPU memory PyTorch held at once (tensors plus its cache); what decides an out-of-memory
  held unused: reserved minus allocated after a phase or part move: memory PyTorch keeps without tensors in it
  frag max: the share of that unused memory outside its largest hole (near 1: only small pieces, fragmentation)
  inactive split: free space inside cache blocks that still hold live tensors (reusable only by what fits the holes)
  alloc retries: the allocator had to free its cache and retry a GPU allocation (memory pressure or fragmentation)
  device mallocs/frees: GPU allocations and releases (hipMalloc/hipFree), each a synchronization
  MemPool reserved: memory in dedicated MemPools (OT_ROCM_MEM_POOLS); returned whole when its part unloads
  GPU temp/clock: read when amdsmi/pynvml is installed; a hotter, slower-clocked run has slower steps for that reason"""


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3g}" if abs(v) < 1000 else f"{v:.0f}"
    return str(v)


def _table(rows: list[list[str]]) -> str:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip() for r in rows)


def _loss_agreement(a: Run, b: Run) -> str:
    la = {s["step"]: s["loss"] for s in a.of("step")}
    lb = {s["step"]: s["loss"] for s in b.of("step")}
    common = sorted(set(la) & set(lb))
    if not common:
        return "no common steps"
    differ = [s for s in common if la[s] != lb[s]]
    if not differ:
        return f"identical losses over {len(common)} steps"
    worst = max(abs(la[s] - lb[s]) / max(abs(la[s]), 1e-12) for s in common)
    return f"losses differ from step {differ[0]} on ({len(differ)} of {len(common)} steps), max rel diff {worst:.2e}"


def _weights_agreement(a: Run, b: Run, compare_weights: bool) -> str:
    oa, ob = summarize(a)["output"], summarize(b)["output"]
    if not oa or not ob:
        return "no saved model in one of them"
    if oa["digest"] == ob["digest"]:
        return "bit-identical saved weights"
    if not compare_weights:
        return "saved weights differ"
    try:
        return "saved weights differ: " + _weights_difference(oa["path"], ob["path"])
    except Exception as e:
        return f"saved weights differ (not compared: {type(e).__name__}: {e})"


def _weights_difference(path_a: str, path_b: str) -> str:
    from safetensors import safe_open

    def files(path):
        if os.path.isdir(path):
            return {os.path.relpath(os.path.join(r, n), path): os.path.join(r, n)
                    for r, _, ns in os.walk(path) for n in ns if n.endswith(".safetensors")}
        return {"": path}

    fa, fb = files(path_a), files(path_b)
    worst, worst_key = 0.0, ""
    for name in sorted(set(fa) & set(fb)):
        with safe_open(fa[name], framework="pt") as a, safe_open(fb[name], framework="pt") as b:
            for key in sorted(set(a.keys()) & set(b.keys())):
                ta, tb = a.get_tensor(key).double(), b.get_tensor(key).double()
                if ta.shape != tb.shape:
                    return f"{key}: shapes {tuple(ta.shape)} vs {tuple(tb.shape)}"
                err = ((ta - tb).norm() / ta.norm().clamp_min(1e-30)).item() if ta.numel() else 0.0
                if err > worst:
                    worst, worst_key = err, key
    return f"largest relative difference {worst:.2e} ({worst_key})" if worst_key else "no common tensors"


def report(paths: list[str], compare_weights: bool = True) -> tuple[str, dict]:
    runs = [load(p) for p in paths]
    summaries = [summarize(r) for r in runs]
    lines = []
    labels = [r.label for r in runs]

    # settings that differ (the point of an A/B), without the per-run folders
    flat = [settings(r) for r in runs]
    keys = sorted(set().union(*flat))
    ignored = (*_IGNORED_KEYS, *(f"config.{k}" for k in _IGNORED_KEYS))
    differing = [k for k in keys
                 if len({json.dumps(f.get(k), default=str) for f in flat}) > 1 and not k.endswith("uuid")
                 and k not in ignored and not k.startswith(tuple(f"{i}." for i in ignored))]
    lines.append("runs:")
    for r, s in zip(runs, summaries, strict=True):
        h = r.header
        lines.append(f"  {r.label}: {s['status']}, {s['steps']} steps, variant {r.variant}, commit {h.get('commit')}, "
                     f"{h.get('gpu', h.get('device'))}, MemPools {h.get('mem_pools')}  ({r.path})")
    if len(runs) > 1:
        lines.append("")
        lines.append("settings that differ:" if differing else "settings that differ: none")
        if differing:
            rows = [["setting", *labels]] + [[k, *[_fmt(f.get(k)) for f in flat]] for k in differing]
            lines.append(_table(rows))

    lines.append("")
    metric_keys = [k for k in summaries[0] if k not in ("phases", "output", "status")]
    rows = [["", *labels]] + [[k, *[_fmt(s.get(k)) for s in summaries]] for k in metric_keys]
    lines.append(_table(rows))

    phase_names = sorted({n for s in summaries for n in s["phases"]})
    if phase_names:
        lines.append("")
        lines.append("phases (count, total s, peak reserved GiB, most held unused after it GiB):")
        rows = [["phase", *labels]]
        for n in phase_names:
            cells = []
            for s in summaries:
                p = s["phases"].get(n)
                cells.append("-" if p is None else f"{p['count']}x {p['seconds']:.1f}s "
                             f"{_fmt(p['peak reserved GiB'])} {_fmt(p['held unused max GiB'])}")
            rows.append([n, *cells])
        lines.append(_table(rows))

    if len(runs) > 1:
        lines.append("")
        lines.append(f"compared with {labels[0]}:")
        lines.extend(f"  {r.label}: {_loss_agreement(runs[0], r)}; {_weights_agreement(runs[0], r, compare_weights)}"
                     for r in runs[1:])
        variants: dict[str, list[Run]] = {}
        for r in runs:
            variants.setdefault(r.variant, []).append(r)
        repeats = {v: rs for v, rs in variants.items() if len(rs) > 1}
        if repeats:
            lines.append("")
            lines.append("repeats of the same variant (the run-to-run noise to judge the differences above by):")
            lines.extend(f"  {v}: {rs[0].label} vs {r.label}: {_loss_agreement(rs[0], r)}; "
                         f"{_weights_agreement(rs[0], r, compare_weights)}"
                         for v, rs in repeats.items() for r in rs[1:])
        if len(variants) > 1 and repeats:
            lines.append("")
            lines.append("per variant (mean, min-max over its runs):")
            agg_keys = [k for k in metric_keys if all(isinstance(s.get(k), int | float) for s in summaries)]
            rows = [["", *variants]]
            for k in agg_keys:
                cells = []
                for rs in variants.values():
                    vals = [summaries[runs.index(r)][k] for r in rs]
                    cells.append(_fmt(statistics.fmean(vals)) if len(vals) == 1 else
                                 f"{_fmt(statistics.fmean(vals))} ({_fmt(min(vals))}-{_fmt(max(vals))})")
                rows.append([k, *cells])
            lines.append(_table(rows))
    if any("peak reserved GiB" in s for s in summaries):
        lines.append("")
        lines.append(_LEGEND)
    data = {"runs": [{"path": r.path, "label": r.label, "variant": r.variant, **s}
                     for r, s in zip(runs, summaries, strict=True)],
            "differing_settings": differing}
    return "\n".join(lines), data
