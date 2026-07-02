"""Run Experiment 1: naive current-context AI vs structured kNN baselines."""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from aihub.src.llm.models.factory import AIModelFactory
from aihub.src.config import AIHubConfig
from aihub.src.predict.ablation import run_current_context_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/exports/stockmem_records.ndjson",
        help="NDJSON export of stockmem_records with payload records.",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/current_context_ai_eval",
        help="Directory for JSONL predictions and summary artifacts.",
    )
    parser.add_argument(
        "--weights",
        default="stockmem/config/weights.auto.json",
        help="Fixed-kNN weights JSON used by the current StockMem pipeline.",
    )
    parser.add_argument(
        "--fixed-head",
        default="stockmem/config/knn_head.fixed_knn_rolling_stable.json",
        help="Tuned fixed-kNN head config used as the main structured baseline.",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--label-threshold", type=float, default=2.0)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--eval-date-from", default=None)
    parser.add_argument("--eval-date-to", default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--prediction-max-attempts", type=int, default=4)
    parser.add_argument("--prediction-retry-delay-seconds", type=float, default=8.0)
    parser.add_argument("--throttle-seconds", type=float, default=1.5)
    parser.add_argument(
        "--prompt-style",
        default="compact",
        choices=["compact", "legacy"],
    )
    parser.add_argument("--replace-existing-dates", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    config = AIHubConfig()
    factory = AIModelFactory(config)
    llm = factory.get_client(factory.resolve_backend(config.predict_llm_backend))
    result = await run_current_context_evaluation(
        llm=llm,
        data_path=Path(args.data),
        out_dir=Path(args.out_dir),
        weights_path=Path(args.weights),
        fixed_head_path=Path(args.fixed_head),
        k=args.k,
        label_threshold=args.label_threshold,
        max_queries=args.max_queries,
        eval_date_from=args.eval_date_from,
        eval_date_to=args.eval_date_to,
        progress_every=args.progress_every,
        resume=not args.no_resume,
        prediction_max_attempts=args.prediction_max_attempts,
        prediction_retry_delay_seconds=args.prediction_retry_delay_seconds,
        throttle_seconds=args.throttle_seconds,
        prompt_style=args.prompt_style,
        replace_existing_dates=args.replace_existing_dates,
    )
    logging.info("summary json: %s", result["summary_path"])
    logging.info("summary md:   %s", result["markdown_path"])


if __name__ == "__main__":
    asyncio.run(main())
