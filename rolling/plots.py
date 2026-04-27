from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_trade_return_histogram(trades: pd.DataFrame, *, output_path: Path) -> None:
    """Histogram of trade-level returns."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if trades.empty:
        ax.set_title("Trade Return Histogram (no trades)")
    else:
        vals = trades["trade_return"].dropna()
        ax.hist(vals, bins=40, color="tab:blue", alpha=0.8, edgecolor="black")
        ax.set_title("Trade Return Histogram")
        ax.set_xlabel("Trade Return")
        ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_cumulative_return_curve(oos_equity: pd.DataFrame, *, output_path: Path) -> None:
    """Cumulative return curve from stitched OOS equity."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    if oos_equity.empty:
        ax.set_title("Cumulative Return Curve (no OOS data)")
    else:
        base = float(oos_equity["equity"].iloc[0])
        curve = oos_equity["equity"] / base - 1.0
        ax.plot(oos_equity["datetime"], curve, linewidth=1.0, color="tab:green")
        ax.set_title("Cumulative Return Curve")
        ax.set_xlabel("Time")
        ax.set_ylabel("Cumulative Return")
        ax.grid(True, alpha=0.25)
        fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_rolling_sharpe(sharpe: pd.Series, *, output_path: Path) -> None:
    """Rolling Sharpe ratio chart."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    if sharpe.empty:
        ax.set_title("Rolling Sharpe Ratio (no OOS data)")
    else:
        ax.plot(sharpe.index, sharpe.values, linewidth=1.0, color="tab:purple")
        ax.axhline(0.0, linestyle="--", linewidth=0.8, color="black")
        ax.set_title("Rolling Sharpe Ratio")
        ax.set_xlabel("Time")
        ax.set_ylabel("Sharpe")
        ax.grid(True, alpha=0.25)
        fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

