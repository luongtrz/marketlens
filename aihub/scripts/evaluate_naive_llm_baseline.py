"""Official wrapper for the naive current-context LLM baseline."""

from __future__ import annotations

import asyncio

from aihub.scripts.evaluate_prediction_ablation import main


if __name__ == "__main__":
    asyncio.run(main())
