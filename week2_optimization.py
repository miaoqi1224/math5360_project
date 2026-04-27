"""
Week 2 - parameter grid search and sensitivity heatmaps.

Uses PDF full grid (951 x 96) and Group 5 CO contract slippage / point value from
``group5_config`` (optional ``group5_overrides.json``).

Uses existing strategy code from main.py:
    rolling_hh_ll(), run_backtest() (via run_strategy()).

Run from project folder:
    python week2_optimization.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from group5_config import PRIMARY_DATA_FILE, SECONDARY_DATA_FILE, contract_slippage_point_value
from main import (
    load_market_data,
    optimize_parameters,
    plot_optimization_heatmaps,
    week2_interpretation_text,
)


def prepare_ohlc_for_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-positive closes so log-style paths stay finite (same as exploratory hygiene)."""
    out = df.copy()
    ok = out["Close"].to_numpy(dtype=float) > 0.0
    return out.loc[ok]


def run_week2(
    *,
    market: str = "primary",
    data_file: str | None = None,
    out_dir: str | Path | None = None,
    out_csv: str = "optimization_results.csv",
    bars_back: int = 17001,
    fast_grid: bool = False,
) -> pd.DataFrame:
    csv = data_file or (SECONDARY_DATA_FILE if market == "secondary" else PRIMARY_DATA_FILE)
    raw = load_market_data(csv)
    data = prepare_ohlc_for_backtest(raw)

    if fast_grid:
        L_grid = np.arange(500, 3001, 500, dtype=int)
        S_grid = np.arange(0.005, 0.0251, 0.005, dtype=float)
        print("Using --fast-grid (subset); for PDF submission use default without this flag.")
    else:
        L_grid, S_grid = None, None

    slpg_f, pv_f = contract_slippage_point_value(market)
    results = optimize_parameters(
        data,
        L_grid=L_grid,
        S_grid=S_grid,
        bars_back=bars_back,
        slpg=slpg_f,
        pv=pv_f,
        e0=100_000.0,
        verbose=True,
    )

    if out_dir is not None:
        base = Path(out_dir)
    else:
        sub = "week2_btc" if market == "secondary" else "week2"
        base = Path(__file__).resolve().parent / "results" / sub
    base.mkdir(parents=True, exist_ok=True)
    out_path = base / out_csv
    # Required columns for the assignment table; keep extras for research
    cols_core = ["L", "S", "return", "max_dd", "return_to_dd", "sharpe", "trades"]
    results_sorted = results.sort_values("return_to_dd", ascending=False, na_position="last")
    results_sorted[cols_core].to_csv(out_path, index=False, float_format="%.8g")
    results.to_csv(
        out_path.with_name("optimization_results_full.csv"),
        index=False,
        float_format="%.8g",
    )

    scores = results["return_to_dd"].astype(float).replace([np.inf, -np.inf], np.nan)
    best_idx = int(scores.fillna(-np.inf).idxmax())
    best = results.loc[best_idx]

    print("\n" + "=" * 72)
    print("BEST PARAMETERS (maximize return_to_dd = return / abs(max_dd))")
    print("=" * 72)
    print(f"  best_L = {int(best['L'])}")
    print(f"  best_S = {float(best['S']):.4f}")
    print("\n  Metrics:")
    for k in ("return", "max_dd", "return_to_dd", "sharpe", "trades", "win_rate", "avg_trade_return"):
        if k in best.index:
            print(f"    {k}: {best[k]}")

    print("\n" + "=" * 72)
    print("TOP 5 (L, S) BY return_to_dd")
    print("=" * 72)
    top5 = results.sort_values("return_to_dd", ascending=False, na_position="last").head(5)
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(top5.to_string(index=False))

    prefix = str(out_path.with_name("optimization"))
    fig1, fig2 = plot_optimization_heatmaps(results, save_prefix=prefix)
    print(f"\nHeatmaps saved: {prefix}_heatmap_return_to_dd.png , {prefix}_heatmap_sharpe.png")

    interp = week2_interpretation_text(results, best)
    print("\n" + interp)

    backend = plt.matplotlib.get_backend().lower()
    if "agg" not in backend and os.environ.get("MPLBACKEND", "").lower() != "agg":
        plt.show()

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Week 2 grid search (primary CO or secondary BTC).")
    ap.add_argument(
        "--market",
        choices=("primary", "secondary"),
        default="primary",
        help="Contract economics from group5_config / overrides for chosen market.",
    )
    ap.add_argument("--data", default=None, help="Override CSV path.")
    ap.add_argument("--out-dir", default=None, help="Output directory (default: results/week2 or week2_btc).")
    ap.add_argument(
        "--fast-grid",
        action="store_true",
        help="Small L/S grid for development (default = PDF full 951×96 grid).",
    )
    args = ap.parse_args()
    run_week2(
        market=args.market,
        data_file=args.data,
        out_dir=args.out_dir,
        fast_grid=args.fast_grid,
    )
