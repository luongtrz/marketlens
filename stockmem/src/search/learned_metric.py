from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .embedder import SplitEmbedding


_EPS = 1e-12


def normalize_block(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm <= _EPS:
        return np.zeros_like(arr)
    return arr / norm


@dataclass(frozen=True)
class LearnedDiagonalMetric:
    block_dims: tuple[int, ...]
    diagonal: np.ndarray
    block_scales: np.ndarray
    version: str = "learned_diagonal_v1"

    def __post_init__(self) -> None:
        if len(self.block_dims) not in {3, 4}:
            raise ValueError("Learned metric must contain three or four feature blocks")
        if not self.block_dims or any(dim <= 0 for dim in self.block_dims):
            raise ValueError("Learned metric block dimensions must be positive")
        if sum(self.block_dims) != self.diagonal.size:
            raise ValueError("Learned metric diagonal does not match block dimensions")
        if len(self.block_dims) != self.block_scales.size:
            raise ValueError("Learned metric scales do not match block dimensions")
        if not np.all(np.isfinite(self.diagonal)) or not np.all(
            np.isfinite(self.block_scales)
        ):
            raise ValueError("Learned metric parameters must be finite")
        if np.any(self.diagonal < 0) or np.any(self.block_scales < 0):
            raise ValueError("Learned metric parameters must be non-negative")
        if float(self.block_scales.sum()) <= _EPS:
            raise ValueError("Learned metric must have at least one positive block scale")
        offset = 0
        for dim, scale in zip(self.block_dims, self.block_scales):
            if scale > _EPS and not np.any(self.diagonal[offset : offset + dim] > _EPS):
                raise ValueError("Each active learned metric block must have a positive diagonal")
            offset += dim

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> LearnedDiagonalMetric:
        metric_type = payload.get("type")
        if metric_type not in {"learned_diagonal", "learned_linear"}:
            raise ValueError(f"Unsupported learned retriever type: {metric_type}")
        block_dims = tuple(int(v) for v in payload["block_dims"])
        diagonal = np.asarray(payload["d"], dtype=np.float64)
        scales = np.asarray(payload["block_scales"], dtype=np.float64)
        protocol = payload.get("mining_protocol", {})
        version = str(
            payload.get("version")
            or (
                protocol.get("version")
                if isinstance(protocol, dict)
                else None
            )
            or "learned_diagonal_v1"
        )
        return cls(
            block_dims=block_dims,
            diagonal=diagonal,
            block_scales=scales,
            version=version,
        )

    @classmethod
    def load(cls, path: str | Path) -> LearnedDiagonalMetric:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Learned retriever artifact must contain a JSON object")
        return cls.from_payload(payload)

    def score_batch(
        self,
        query_blocks: Sequence[np.ndarray],
        cand_stacked: Sequence[np.ndarray],
    ) -> np.ndarray:
        """Score one query against N candidates in vectorized form.

        cand_stacked: one array per block, each shaped (N, dim_b).
        Returns: float64 array of shape (N,).
        """
        if len(query_blocks) != len(self.block_dims) or len(cand_stacked) != len(self.block_dims):
            raise ValueError("Block count mismatch in score_batch")
        N = cand_stacked[0].shape[0]
        total = np.zeros(N, dtype=np.float64)
        offset = 0
        for b, (dim, scale) in enumerate(zip(self.block_dims, self.block_scales)):
            d_b = self.diagonal[offset : offset + dim]
            offset += dim
            if float(scale) <= _EPS:
                continue
            q = np.asarray(query_blocks[b], dtype=np.float64) * d_b
            qn = float(np.linalg.norm(q))
            if qn <= _EPS:
                continue
            q_hat = q / qn
            C = np.asarray(cand_stacked[b], dtype=np.float64) * d_b[None, :]
            C_norms = np.linalg.norm(C, axis=1, keepdims=True)
            C_hat = np.where(C_norms > _EPS, C / np.maximum(C_norms, _EPS), 0.0)
            total += float(scale) * (C_hat @ q_hat)
        return total

    def score(self, query_blocks: Sequence[np.ndarray], candidate_blocks: Sequence[np.ndarray]) -> float:
        if len(query_blocks) != len(self.block_dims) or len(candidate_blocks) != len(self.block_dims):
            raise ValueError("Input blocks do not match learned metric")

        offset = 0
        score = 0.0
        for dim, scale, query, candidate in zip(
            self.block_dims, self.block_scales, query_blocks, candidate_blocks
        ):
            q = np.asarray(query, dtype=np.float64)
            c = np.asarray(candidate, dtype=np.float64)
            if q.size != dim or c.size != dim:
                raise ValueError("Input vector dimension does not match learned metric")
            block_d = self.diagonal[offset : offset + dim]
            q_weighted = normalize_block(block_d * q)
            c_weighted = normalize_block(block_d * c)
            score += float(scale) * float(np.dot(q_weighted, c_weighted))
            offset += dim
        return score

    def score_split(self, query: SplitEmbedding, candidate: SplitEmbedding) -> float:
        if len(self.block_dims) == 4:
            query_blocks = (
                query.event_vec,
                query.factor_vec,
                query.indicator_vec,
                query.price_vec,
            )
            candidate_blocks = (
                candidate.event_vec,
                candidate.factor_vec,
                candidate.indicator_vec,
                candidate.price_vec,
            )
        else:
            query_blocks = (query.factor_vec, query.indicator_vec, query.price_vec)
            candidate_blocks = (
                candidate.factor_vec,
                candidate.indicator_vec,
                candidate.price_vec,
            )
        return self.score(
            query_blocks,
            candidate_blocks,
        )


def load_learned_metric(path: str | Path | None) -> LearnedDiagonalMetric | None:
    if not path:
        return None
    artifact_path = Path(path)
    if not artifact_path.exists():
        return None
    if not artifact_path.read_text(encoding="utf-8").strip():
        return None
    return LearnedDiagonalMetric.load(artifact_path)
