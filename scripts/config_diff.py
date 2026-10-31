# Compares two OneTrainer config files the way training loads them (migrated, on top of the defaults and the
# optimizer's UI defaults) and prints every setting that differs, including ones the UI hides.
# usage: ./run-cmd.sh config_diff old.json new.json
from util.import_util import script_imports

script_imports()

import json
import os
import sys

from modules.util.config.TrainConfig import TrainConfig
from modules.util.optimizer_util import default_optimizer_config


def load(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    builtin = os.path.basename(path).startswith("#") and os.path.basename(path) != "#.json"
    optimizer = TrainConfig.default_values().from_dict(json.loads(json.dumps(data)), migrate=not builtin).optimizer.optimizer
    config = TrainConfig.default_values()
    config.optimizer = default_optimizer_config(optimizer)
    config.from_dict(data, migrate=not builtin)
    print(f"{path}: __version {data.get('__version')}{' (built-in name: not migrated)' if builtin else ''}")
    return config.to_dict()

def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and key not in ("optimizer_defaults",):
            out.update(flat(v, key + "."))
        else:
            out[key] = v
    return out

a, b = flat(load(sys.argv[1])), flat(load(sys.argv[2]))
skip = ("concepts", "samples", "secrets", "optimizer_defaults", "run_name", "__version", "embedding.uuid")
rows = [(k, a.get(k, "<missing>"), b.get(k, "<missing>")) for k in sorted(set(a) | set(b))
        if a.get(k) != b.get(k) and not k.startswith(skip)]
print(f"\n{len(rows)} differing settings ({os.path.basename(sys.argv[1])}  vs  {os.path.basename(sys.argv[2])}):")
for k, x, y in rows:
    print(f"  {k}: {json.dumps(x)[:70]}  vs  {json.dumps(y)[:70]}")
