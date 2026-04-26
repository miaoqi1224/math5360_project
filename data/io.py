from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Week4Paths:
    """All input/output paths used by the Week 4 analysis pipeline."""

    market_data: Path
    oos_equity: Path
    rolling_parameters: Path
    oos_trades: Path
    output_dir: Path


def load_week4_inputs(paths: Week4Paths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load market and week-3 output tables with deterministic parsing."""
    market = pd.read_csv(paths.market_data)
    oos_equity = pd.read_csv(paths.oos_equity, parse_dates=["datetime"])
    rolling_params = pd.read_csv(
        paths.rolling_parameters,
        parse_dates=["train_start", "train_end", "oos_start", "oos_end"],
    )
    oos_trades = pd.read_csv(paths.oos_trades, parse_dates=["entry_time", "exit_time"])
    return market, oos_equity, rolling_params, oos_trades


def ensure_output_dir(path: Path) -> None:
    """Create output directory tree if missing."""
    path.mkdir(parents=True, exist_ok=True)

