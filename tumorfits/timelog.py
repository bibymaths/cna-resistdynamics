# SPDX-FileCopyrightText: 2025 Abhinav Mishra
# SPDX-License-Identifier: MIT
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

# Enable rich tracebacks globally
install(show_locals=False)

# Shared console (prevents multiple instances)
_console = Console()


def get_logger(name: str = "tumorfit", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)

    handler = RichHandler(
        console=_console,
        rich_tracebacks=True,
        show_time=True,
        show_level=True,
        show_path=False,  # cleaner; set True if you want file:line
        markup=True,
    )

    formatter = logging.Formatter(
        fmt="%(message)s",  # Rich handles time/level formatting
        datefmt="[%H:%M:%S]",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False

    return logger


@dataclass
class Timer:
    t0: float = None  # type: ignore

    def __post_init__(self) -> None:
        self.t0 = time.time()

    def s(self) -> float:
        return time.time() - self.t0


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")