"""Load the YAML pipeline configuration into attribute-style dictionaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigNode(dict):
    """A dict whose keys can also be read as attributes (cfg.detection.model)."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return ConfigNode({k: _wrap(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def load_yaml(path: str | Path) -> ConfigNode:
    with open(path, encoding="utf-8") as f:
        return _wrap(yaml.safe_load(f) or {})


def load_config(path: str | Path = "config/default.yaml", overrides: dict | None = None) -> ConfigNode:
    cfg = load_yaml(path)
    for dotted_key, value in (overrides or {}).items():
        node = cfg
        *parents, leaf = dotted_key.split(".")
        for key in parents:
            node = node.setdefault(key, ConfigNode())
        node[leaf] = value
    return cfg
