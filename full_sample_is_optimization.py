"""
Full-sample in-sample parameter search (PDF optional baseline for IS vs OOS decay).

Runs one ``optimize_parameters`` pass on the entire cleaned history (no walk-forward).
Does not change core trading logic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from group5_config import PRIMARY_DATA_FILE, SECONDARY_DATA_FILE, contract_slippage_point_value
from main import compute_metrics, load_market_data, optimize_parameters, run_strategy


def main() -> None:
    p = argparse.ArgumentParser(description="Full-history IS grid search (single best (L,S) report row).")
    p.add_argument("--market", choices=("primary", "secondary"), default="primary")
    p.add_argument("--data", default=None, help="Override CSV path.")
    p.add_argument(
        "--fast-grid",
        action="store_true",
        help="Small L/S grid for smoke tests (default: PDF full 951×96).",
    )
    p.add_argument(
        "--out-csv",
        default=None,
        help="Output CSV path (default: results/is_full_sample_<market>.csv).",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parent
    csv = args.data or (SECONDARY_DATA_FILE if args.market == "secondary" else PRIMARY_DATA_FILE)
    out_csv = Path(args.out_csv or root / "results" / f"is_full_sample_{args.market}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    raw = load_market_data(csv)
    ok = raw["Close"].to_numpy(dtype=float) > 0.0
    data = raw.loc[ok].reset_index(drop=True)

    if args.fast_grid:
        L_grid = np.arange(500, 3001, 500, dtype=int)
        S_grid = np.arange(0.005, 0.0251, 0.005, dtype=float)
        print("Using --fast-grid (subset).")
    else:
        L_grid, S_grid = None, None

    slpg_f, pv_f = contract_slippage_point_value(args.market)
    grid = optimize_parameters(
        data,
        L_grid=L_grid,
        S_grid=S_grid,
        slpg=slpg_f,
        pv=pv_f,
        e0=100_000.0,
        verbose=True,
    )
    scores = grid["return_to_dd"].astype(float).replace([np.inf, -np.inf], np.nan)
    best_idx = int(scores.fillna(-np.inf).idxmax())
    best = grid.loc[best_idx]
    L_best, S_best = int(best["L"]), float(best["S"])

    out = run_strategy(data, L_best, S_best, slpg=slpg_f, pv=pv_f)
    m = compute_metrics(out["E"], out["DD"], out["trades"], out["pnl"])

    row = {
        "market": args.market,
        "data_file": str(csv),
        "best_L": L_best,
        "best_S": S_best,
        "grid_best_return_to_dd": float(best["return_to_dd"]),
        "fullsample_total_return": m["total_return"],
        "fullsample_max_drawdown": m["max_drawdown"],
        "fullsample_sharpe_ratio": m["sharpe_ratio"],
        "fullsample_total_trades": m["total_trades"],
    }
    pd.DataFrame([row]).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
