"""This module reads configuration and refuses implicit monetary defaults."""
from pathlib import Path

import yaml

from cycletime.contracts import JSON


def load_config(path: Path) -> JSON:
    """This function must load a YAML mapping; it must refuse missing files, empty roots, and non-string keys."""
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict) or not all(isinstance(key, str) for key in data):
        raise ValueError(f"{path} must contain a mapping with string keys")
    return dict(data)
