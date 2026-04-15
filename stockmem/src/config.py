from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path

import yaml


def _as_float(value: object, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


@dataclass(frozen=True)
class SearchWeights:
    w1_factor: float
    w2_indicator: float
    w3_price: float


def _normalize_weights(w1: float, w2: float, w3: float) -> SearchWeights:
    w1_pos = max(0.0, w1)
    w2_pos = max(0.0, w2)
    w3_pos = max(0.0, w3)
    total = w1_pos + w2_pos + w3_pos
    if total <= 1e-12:
        return SearchWeights(0.35, 0.2, 0.45)
    return SearchWeights(w1_pos / total, w2_pos / total, w3_pos / total)


def load_weights_from_config(
    config_file: str | None = None,
    weights_file: str | None = None,
) -> SearchWeights:
    root_dir = Path(__file__).resolve().parents[1]
    default_config = root_dir / "config.yaml"
    cfg_path = Path(config_file) if config_file else default_config

    w1 = 0.35
    w2 = 0.2
    w3 = 0.45

    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        search_cfg = cfg.get("search", {}) if isinstance(cfg, dict) else {}
        weights_cfg = search_cfg.get("weights", {}) if isinstance(search_cfg, dict) else {}
        if isinstance(weights_cfg, dict):
            w1 = _as_float(weights_cfg.get("w1_factor"), w1)
            w2 = _as_float(weights_cfg.get("w2_indicator"), w2)
            w3 = _as_float(weights_cfg.get("w3_price"), w3)

    if weights_file:
        wf = Path(weights_file)
        if wf.exists():
            with wf.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            w1 = _as_float(payload.get("w1_factor"), w1)
            w2 = _as_float(payload.get("w2_indicator"), w2)
            w3 = _as_float(payload.get("w3_price"), w3)

    w1 = _as_float(os.getenv("W1_FACTOR"), w1)
    w2 = _as_float(os.getenv("W2_INDICATOR"), w2)
    w3 = _as_float(os.getenv("W3_PRICE"), w3)
    return _normalize_weights(w1, w2, w3)


@dataclass(frozen=True)
class Settings:
    vector_backend: str = os.getenv("VECTOR_BACKEND", "memory")
    db_url: str = os.getenv("DB_URL", "sqlite+aiosqlite:///test.db")
    weights: SearchWeights = load_weights_from_config(
        os.getenv("STOCKMEM_CONFIG"),
        os.getenv("WEIGHTS_FILE"),
    )


settings = Settings()


def sqlite_path_from_url(db_url: str) -> str:
    prefix_async = "sqlite+aiosqlite:///"
    prefix_sync = "sqlite:///"
    if db_url.startswith(prefix_async):
        return db_url[len(prefix_async):]
    if db_url.startswith(prefix_sync):
        return db_url[len(prefix_sync):]
    raise ValueError(
        f"Unsupported DB_URL: {db_url}. Expected sqlite+aiosqlite:///path.db"
    )
