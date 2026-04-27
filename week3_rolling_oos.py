"""
Week 3: rolling walk-forward OOS backtest (calls main.rolling_backtest).

Saves oos_equity.csv, rolling_parameters.csv, oos_trades.csv (OOS round-turns),
and optional equity/DD figures under results/week3/ by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from group5_config import PRIMARY_DATA_FILE, SECONDARY_DATA_FILE, contract_slippage_point_value
from main import load_market_data, plot_rolling_oos, rolling_backtest


def _fast_grids() -> tuple[np.ndarray, np.ndarray]:
    """Small grid for smoke tests only (not the assignment PDF grid)."""
    L_grid = np.arange(500, 3001, 500, dtype=int)
    S_grid = np.arange(0.01, 0.0301, 0.01, dtype=float)
    return L_grid, S_grid


def main() -> None:
    p = argparse.ArgumentParser(description="Rolling OOS walk-forward backtest (Week 3).")
    p.add_argument(
        "--market",
        choices=("primary", "secondary"),
        default="primary",
        help="primary=CO, secondary=BTC (see group5_config data file names).",
    )
    p.add_argument(
        "--data",
        type=str,
        default=None,
        help="Override CSV path (default: CO.csv or BTC-5minHLV.csv from config).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory for CSVs and figures (default: results/week3 or results/week3_btc).",
    )
    p.add_argument(
        "--fast-grid",
        action="store_true",
        help="Use a small L/S grid for debugging (default = PDF full grid per window, very slow).",
    )
    p.add_argument(
        "--max-segments",
        type=int,
        default=None,
        help="Stop after this many OOS windows (debug / smoke test).",
    )
    p.add_argument(
        "--is-years",
        type=float,
        default=4.0,
        help="In-sample window length in years (T). Default 4.0 matches PDF-style setup.",
    )
    p.add_argument(
        "--oos-months",
        type=float,
        default=3.0,
        help="Out-of-sample horizon in months (τ). Default 3.0 matches PDF-style setup.",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Do not call plt.show() (still saves PNG if --save-fig).",
    )
    p.add_argument(
        "--save-fig",
        action="store_true",
        help="Save rolling_oos_equity_dd.png under --out-dir.",
    )
    args = p.parse_args()

    data_path = args.data or (SECONDARY_DATA_FILE if args.market == "secondary" else PRIMARY_DATA_FILE)
    out_dir = Path(
        args.out_dir
        or ("results/week3_btc" if args.market == "secondary" else "results/week3")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_market_data(data_path)
    if args.fast_grid:
        L_grid, S_grid = _fast_grids()
    else:
        L_grid, S_grid = None, None

    slpg_f, pv_f = contract_slippage_point_value(args.market)
    out = rolling_backtest(
        df,
        is_years=float(args.is_years),
        oos_months=float(args.oos_months),
        L_grid=L_grid,
        S_grid=S_grid,
        max_segments=args.max_segments,
        slpg=slpg_f,
        pv=pv_f,
    )

    oos_path = out_dir / "oos_equity.csv"
    param_path = out_dir / "rolling_parameters.csv"
    trades_path = out_dir / "oos_trades.csv"
    out["oos_equity"].to_csv(oos_path, index=False)
    out["rolling_parameters"].to_csv(param_path, index=False)
    out["oos_trades"].to_csv(trades_path, index=False)

    print(f"Saved {oos_path} ({len(out['oos_equity'])} OOS rows)")
    print(f"Saved {param_path} ({len(out['rolling_parameters'])} windows)")
    print(f"Saved {trades_path} ({len(out['oos_trades'])} completed trades)")
    print(
        f"global_return={out['global_return']:.2f}, "
        f"global_max_drawdown={out['global_max_drawdown']:.2f}, "
        f"n_segments={out['n_segments']}, is_bars={out['is_bars']}, oos_bars={out['oos_bars']}, "
        f"bars_per_trading_day={out['bars_per_trading_day']:.2f}"
    )

    title = "Rolling OOS (BTC)" if args.market == "secondary" else "Rolling OOS (CO)"
    fig = plot_rolling_oos(out["oos_equity"], title_prefix=title)
    if args.save_fig:
        fig_path = out_dir / "rolling_oos_equity_dd.png"
        fig.savefig(fig_path, dpi=150)
        print(f"Saved {fig_path}")
    if not args.no_show:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
