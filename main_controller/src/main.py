"""MainController entry point."""

import logging
import uvicorn

from main_controller.src.api import app

logger = logging.getLogger(__name__)


def main() -> None:
    """Start the MainController service."""
    logger.info("Starting MainController on port 8005")
    uvicorn.run(app, host="0.0.0.0", port=8005)


if __name__ == "__main__":
    main()
