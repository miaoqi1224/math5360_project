from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerfConfig:
    """Configuration for annualization and rolling Sharpe window."""

    bars_per_year: float
    rolling_sharpe_window: int


def build_oos_pnl(oos_equity: pd.DataFrame) -> pd.Series:
    """Bar-level OOS PnL series from stitched equity."""
    if oos_equity.empty:
        return pd.Series(dtype=float, name="pnl")
    pnl = oos_equity["equity"].diff().fillna(0.0).astype(float)
    pnl.index = pd.DatetimeIndex(oos_equity["datetime"])
    pnl.name = "pnl"
    return pnl


def rolling_sharpe_series(pnl: pd.Series, *, window: int, bars_per_year: float) -> pd.Series:
    """Annualized rolling Sharpe on bar-level PnL."""
    if pnl.empty:
        return pd.Series(dtype=float, name="rolling_sharpe")
    if window < 2:
        raise ValueError("rolling_sharpe_window must be >= 2.")
    mean = pnl.rolling(window=window, min_periods=window).mean()
    std = pnl.rolling(window=window, min_periods=window).std(ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = (mean / std) * np.sqrt(float(bars_per_year))
    rs = rs.replace([np.inf, -np.inf], np.nan)
    rs.name = "rolling_sharpe"
    return rs


def compute_performance_summary(
    oos_equity: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    bars_per_year: float,
) -> pd.DataFrame:
    """Compute requested project-level metrics."""
    if oos_equity.empty:
        metrics = {
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "mean_bar_pnl": float("nan"),
            "std_bar_pnl": float("nan"),
            "annualized_mean_bar_pnl": float("nan"),
            "annualized_std_bar_pnl": float("nan"),
            "sharpe_ratio": float("nan"),
            "win_rate": float("nan"),
            "average_win": float("nan"),
            "average_loss": float("nan"),
            "profit_factor": float("nan"),
            "total_number_of_trades": 0.0,
        }
        return pd.DataFrame({"metric": list(metrics.keys()), "value": list(metrics.values())})

    eq = oos_equity["equity"].to_numpy(dtype=float)
    dd = oos_equity["drawdown"].to_numpy(dtype=float)
    pnl_bar = np.diff(eq, prepend=eq[0])

    total_return = float(eq[-1] - eq[0])
    max_drawdown = float(np.min(dd))
    std = float(np.std(pnl_bar, ddof=1)) if len(pnl_bar) > 1 else 0.0
    mean = float(np.mean(pnl_bar))
    # PDF-style "average return" / "std of returns" on the same bar-frequency PnL as Sharpe.
    ann_mean = float(mean * bars_per_year)
    ann_std = float(std * np.sqrt(bars_per_year))
    sharpe = float(mean / std * np.sqrt(bars_per_year)) if std > 0 else float("nan")

    trade_pnl = trades["pnl"].to_numpy(dtype=float) if not trades.empty else np.array([], dtype=float)
    wins = trade_pnl[trade_pnl > 0.0]
    losses = trade_pnl[trade_pnl < 0.0]
    n_trades = float(len(trade_pnl))
    win_rate = float((trade_pnl > 0.0).mean()) if n_trades else float("nan")
    avg_win = float(np.mean(wins)) if wins.size else float("nan")
    avg_loss = float(np.mean(losses)) if losses.size else float("nan")
    profit_factor = float(np.sum(wins) / abs(np.sum(losses))) if losses.size else (float("inf") if wins.size else float("nan"))

    metrics = {
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "mean_bar_pnl": mean,
        "std_bar_pnl": std,
        "annualized_mean_bar_pnl": ann_mean,
        "annualized_std_bar_pnl": ann_std,
        "sharpe_ratio": sharpe,
        "win_rate": win_rate,
        "average_win": avg_win,
        "average_loss": avg_loss,
        "profit_factor": profit_factor,
        "total_number_of_trades": n_trades,
    }
    return pd.DataFrame({"metric": list(metrics.keys()), "value": list(metrics.values())})

