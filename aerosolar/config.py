"""Configuration loading.

All physical parameters live in ``config/*.yaml``; code never hard-codes design values.
A :class:`Config` bundles the four files and supports deep-merged overrides (used by scenario presets
and by the simulator-control UI).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
CONFIG_FILES = ("turbine", "site", "sensors", "scenarios")


def deep_merge(base: dict, override: dict | None) -> dict:
    """Return a copy of ``base`` with ``override`` merged in recursively (override wins)."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass
class Config:
    """The four configuration trees: turbine, site, sensors, scenarios."""

    turbine: dict
    site: dict
    sensors: dict
    scenarios: dict
    source_dir: Path = field(default=CONFIG_DIR)

    @classmethod
    def load(cls, config_dir: Path | str | None = None) -> "Config":
        cdir = Path(config_dir) if config_dir else CONFIG_DIR
        trees = {name: load_yaml(cdir / f"{name}.yaml") for name in CONFIG_FILES}
        return cls(**trees, source_dir=cdir)

    def with_overrides(self, turbine: dict | None = None, site: dict | None = None,
                       sensors: dict | None = None, scenarios: dict | None = None) -> "Config":
        return Config(turbine=deep_merge(self.turbine, turbine), site=deep_merge(self.site, site),
                      sensors=deep_merge(self.sensors, sensors), scenarios=deep_merge(self.scenarios, scenarios),
                      source_dir=self.source_dir)

    def preset(self, name: str) -> dict:
        presets = self.scenarios.get("presets", {})
        if name not in presets:
            raise KeyError(f"unknown preset {name!r}; available: {sorted(presets)}")
        return copy.deepcopy(presets[name])

    def to_dict(self) -> dict[str, Any]:
        return {"turbine": self.turbine, "site": self.site, "sensors": self.sensors, "scenarios": self.scenarios}
