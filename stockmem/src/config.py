from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from .search.learned_metric import LearnedDiagonalMetric

def _as_float(value: object, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _as_bool(value: object, fallback: bool) -> bool:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
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
            src = payload
            if isinstance(payload, dict) and isinstance(payload.get("weights"), dict):
                src = payload.get("weights")
            if isinstance(src, dict):
                w1 = _as_float(src.get("w1_factor"), w1)
                w2 = _as_float(src.get("w2_indicator"), w2)
                w3 = _as_float(src.get("w3_price"), w3)

    w1 = _as_float(os.getenv("W1_FACTOR"), w1)
    w2 = _as_float(os.getenv("W2_INDICATOR"), w2)
    w3 = _as_float(os.getenv("W3_PRICE"), w3)
    return _normalize_weights(w1, w2, w3)


def load_learned_retriever_from_config(
    artifact_file: str | None = None,
) -> LearnedDiagonalMetric | None:
    from .search.learned_metric import load_learned_metric

    return load_learned_metric(artifact_file)


@dataclass(frozen=True)
class Settings:
    vector_backend: str = os.getenv("VECTOR_BACKEND", "memory")
    db_url: str = os.getenv("DB_URL", "postgresql+asyncpg://postgres:pass@localhost:5432/postgres")
    auto_optimize_enabled: bool = _as_bool(os.getenv("AUTO_OPTIMIZE_ENABLED"), False)
    auto_optimize_hour_utc: int = int(os.getenv("AUTO_OPTIMIZE_HOUR_UTC", "1"))
    auto_optimize_minute_utc: int = int(os.getenv("AUTO_OPTIMIZE_MINUTE_UTC", "15"))
    auto_optimize_horizon: str = os.getenv("AUTO_OPTIMIZE_HORIZON", "7d")
    auto_optimize_trials: int = int(os.getenv("AUTO_OPTIMIZE_TRIALS", "80"))
    auto_optimize_k: int = int(os.getenv("AUTO_OPTIMIZE_K", "5"))
    auto_optimize_warmup: int = int(os.getenv("AUTO_OPTIMIZE_WARMUP", "250"))
    auto_optimize_min_records: int = int(os.getenv("AUTO_OPTIMIZE_MIN_RECORDS", "320"))
    auto_optimize_output: str = os.getenv(
        "AUTO_OPTIMIZE_OUTPUT",
        str(Path(__file__).resolve().parents[1] / "config" / "weights.auto.json"),
    )
    weights: SearchWeights = load_weights_from_config(
        os.getenv("STOCKMEM_CONFIG"),
        os.getenv("WEIGHTS_FILE"),
    )
    learned_retriever_file: str = os.getenv(
        "LEARNED_RETRIEVER_FILE",
        str(Path(__file__).resolve().parents[1] / "config" / "learned_retriever.json"),
    )


settings = Settings()


