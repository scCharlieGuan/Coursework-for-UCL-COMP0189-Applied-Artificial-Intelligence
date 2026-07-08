"""Configuration utilities for the patient readmission project."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def find_project_root(start: Path | None = None) -> Path:
    """Return the nearest parent folder containing configs/config.yaml."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "configs" / "config.yaml").exists():
            return candidate
    raise FileNotFoundError("Could not find project root containing configs/config.yaml")


def load_config(config_path: str | Path = "configs/config.yaml") -> Dict[str, Any]:
    """Load the YAML configuration file."""
    path = Path(config_path)
    if not path.is_absolute():
        path = find_project_root() / path
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def resolve_path(project_root: Path, relative_path: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    path = Path(relative_path)
    return path if path.is_absolute() else project_root / path


def ensure_dirs(*paths: Path) -> None:
    """Create directories if they do not already exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
