"""Shared config loader for train.py and generate.py.

Single source of truth for settings that must stay consistent across
scripts (held-out split size/seed, severity distribution) -- avoids the
failure mode where train.py and generate.py each hardcode their own
defaults and silently drift out of sync with each other or with the real
dataset's actual distribution.
"""

from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = "project_config.yaml"

# Fallback defaults, used only if no config file is found at all -- keeps
# both scripts runnable without requiring project_config.yaml to exist,
# while still preferring the config file when present.
_FALLBACK_CONFIG = {
    "held_out": {"size": 30, "seed": 42},
    "severity_distribution": {"Healthy": 0.25, "Warning": 0.40, "Damaged": 0.35},
    "total_records": {"n": 500, "seed": 100},
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. A partial override of a nested
    key (e.g. only held_out.seed) must not silently drop sibling keys
    (held_out.size) -- a shallow {**base, **override} would do exactly that."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | None = None) -> dict:
    """Load project_config.yaml (or the given path). Missing file falls back
    to _FALLBACK_CONFIG rather than raising -- config is a convenience for
    consistency, not a hard requirement to run either script."""
    config_path = Path(path or DEFAULT_CONFIG_PATH)
    if not config_path.exists():
        return _FALLBACK_CONFIG

    with open(config_path, encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}

    return _deep_merge(_FALLBACK_CONFIG, loaded)
