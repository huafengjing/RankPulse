from __future__ import annotations

from itertools import product
from pathlib import Path

import yaml


def load_parameter_grid(path: str | Path) -> list[dict]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    keys = list(data.keys())
    values = [data[key] for key in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]
