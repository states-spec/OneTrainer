"""Runs one training config under several variants (environment variables and/or config values), each run with its own
workspace, cache and output folder and its own OT_DIAG_LOG, in alternating order, then compares them (diag_report).
The default variants are the MemPool A/B on ROCm: baseline, and OT_ROCM_MEM_POOLS=true."""
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field


@dataclass
class Variant:
    name: str
    env: dict[str, str] = field(default_factory=dict)
    config_values: list[str] = field(default_factory=list)


def parse_variant(spec: str) -> Variant:
    """'NAME env:KEY=VALUE config:key=value ...'"""
    tokens = shlex.split(spec)
    if not tokens or "=" in tokens[0] or ":" in tokens[0]:
        raise ValueError(f"a variant starts with its name: {spec!r}")
    variant = Variant(tokens[0])
    for token in tokens[1:]:
        kind, _, assignment = token.partition(":")
        if "=" not in assignment or kind not in ("env", "config"):
            raise ValueError(f"expected env:KEY=VALUE or config:key=value, got {token!r} in {spec!r}")
        if kind == "env":
            key, _, value = assignment.partition("=")
            variant.env[key] = value
        else:
            variant.config_values.append(assignment)
    return variant


DEFAULT_VARIANTS = ["baseline", "mempools env:OT_ROCM_MEM_POOLS=true"]

# (config key, the key holding the same list inline, TrainConfig's default file)
_REFERENCED_FILES = [("concept_file_name", "concepts", "training_concepts/concepts.json"),
                     ("sample_definition_file_name", "samples", "training_samples/samples.json")]


def missing_referenced_files(config_path: str, variants: list[Variant], config_values: list[str] | None,
                             root: str) -> list[str]:
    """the concepts/samples files the runs would read but that don't exist. Every run copies the config into its
    workspace with these lists inlined (GenericTrainer.__save_config_to_workspace), and fails there without them"""
    with open(config_path) as f:
        config = json.load(f)
    problems = []
    for key, inline_key, default in _REFERENCED_FILES:
        if config.get(inline_key) is not None:
            continue
        for variant in variants:
            path = config.get(key, default)
            for value in [*(config_values or []), *variant.config_values]:
                name, _, assigned = value.partition("=")
                if name == key:
                    path = assigned
            if not os.path.isfile(os.path.join(root, path)):
                problem = f"{key} is {path!r}" + ("" if key in config else " (not in the config: the default)") \
                          + f", which doesn't exist (relative paths start at {root})"
                if problem not in problems:
                    problems.append(problem)
    return problems


def run_ab(config_path: str, variants: list[Variant], repeats: int, out_dir: str, preset_path: str | None = None,
           config_values: list[str] | None = None, python: str = sys.executable) -> list[str]:
    """returns the diagnostics logs, in run order: variant 1, 2, ..., 1, 2, ... (alternating spreads drift over the
    variants: GPU temperature and clocks, page cache, other load)"""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if preset_path is None and (problems := missing_referenced_files(config_path, variants, config_values, root)):
        raise FileNotFoundError("every run would fail before training: " + "; ".join(problems)
                                + ". Point the config (or --config-value) at the files you train with")
    os.makedirs(out_dir, exist_ok=True)
    logs = []
    for repeat in range(1, repeats + 1):
        for variant in variants:
            run_dir = os.path.abspath(os.path.join(out_dir, f"{variant.name}-{repeat}"))
            os.makedirs(run_dir, exist_ok=True)
            log = os.path.join(run_dir, "diag.jsonl")
            if os.path.exists(log):
                os.remove(log)
            cmd = [python, os.path.join(root, "scripts", "train.py"), "--config-path", config_path]
            if preset_path:
                cmd += ["--preset-path", preset_path]
            # own folders, so every run loads, caches and trains the same way; a fixed name, so outputs can be found
            for value in [f"workspace_dir={run_dir}/workspace", f"cache_dir={run_dir}/cache",
                          f"final_output_dir={run_dir}/output", "run_name_mode=CUSTOM", "run_name=model",
                          "continue_last_backup=false", "tensorboard=false", "tensorboard_always_on=false",
                          *(config_values or []), *variant.config_values]:
                cmd += ["--config-value", value]
            env = dict(os.environ)
            # the same LoRA initialization in every run (unseeded otherwise), unless the caller chose a seed
            env.setdefault("OT_SEED", "42")
            env.update(variant.env)
            env["OT_DIAG_LOG"] = log
            env["OT_DIAG_LABEL"] = variant.name
            print(f"=== {variant.name}, run {repeat} of {repeats}: {' '.join(f'{k}={v}' for k, v in variant.env.items())}"
                  f" {' '.join(variant.config_values)}", flush=True)
            start = time.perf_counter()
            with open(os.path.join(run_dir, "output.log"), "w") as out:
                result = subprocess.run(cmd, cwd=root, env=env, stdout=out, stderr=subprocess.STDOUT)
            print(f"    exit code {result.returncode} after {time.perf_counter() - start:.0f} s, log {log}", flush=True)
            if result.returncode != 0:
                with open(os.path.join(run_dir, "output.log"), errors="replace") as f:
                    tail = [line.rstrip() for line in f if line.strip()][-8:]
                print("    the run failed, the end of its output.log:\n      " + "\n      ".join(tail), flush=True)
            logs.append(log)
    return logs
