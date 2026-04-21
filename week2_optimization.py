"""
Week 2 - parameter grid search and sensitivity heatmaps.

Uses existing strategy code from main.py:
    rolling_hh_ll(), run_backtest() (via run_strategy()).

Run from project folder:
    python week2_optimization.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

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
    data_file: str = "HO-5minHLV.csv",
    out_csv: str = "optimization_results.csv",
    bars_back: int = 17001,
) -> pd.DataFrame:
    raw = load_market_data(data_file)
    data = prepare_ohlc_for_backtest(raw)

    results = optimize_parameters(
        data,
        bars_back=bars_back,
        slpg=47.0,
        pv=42000.0,
        e0=100_000.0,
        verbose=True,
    )

    out_path = Path(__file__).resolve().parent / out_csv
    # Required columns for the assignment table; keep extras for research
    cols_core = ["L", "S", "return", "max_dd", "return_to_dd", "sharpe", "trades"]
    results_sorted = results.sort_values("return_to_dd", ascending=False, na_position="last")
    results_sorted[cols_core].to_csv(out_path, index=False, float_format="%.8g")
    results.to_csv(
        out_path.with_name("optimization_results_full.csv"),
        index=False,
        float_format="%.8g",
    )

    best_idx = results["return_to_dd"].idxmax()
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
    run_week2()
