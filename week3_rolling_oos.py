"""
Week 3: rolling walk-forward OOS backtest (calls main.rolling_backtest).

Saves oos_equity.csv, rolling_parameters.csv, and optional equity/DD figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from main import load_market_data, plot_rolling_oos, rolling_backtest


def _default_coarse_grids() -> tuple[np.ndarray, np.ndarray]:
    """Smaller grid for quicker runs; use --full-grid for Week 2 default grids."""
    L_grid = np.arange(500, 3001, 500, dtype=int)
    S_grid = np.arange(0.01, 0.0301, 0.01, dtype=float)
    return L_grid, S_grid


def main() -> None:
    p = argparse.ArgumentParser(description="Rolling OOS walk-forward backtest (Week 3).")
    p.add_argument(
        "--data",
        type=str,
        default="HO-5minHLV.csv",
        help="Path to HO 5-minute CSV.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=".",
        help="Directory for CSVs and figures.",
    )
    p.add_argument(
        "--full-grid",
        action="store_true",
        help="Use main.optimize_parameters default L/S grids (slow per window).",
    )
    p.add_argument(
        "--max-segments",
        type=int,
        default=None,
        help="Stop after this many OOS windows (debug / smoke test).",
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_market_data(args.data)
    if args.full_grid:
        L_grid, S_grid = None, None
    else:
        L_grid, S_grid = _default_coarse_grids()

    out = rolling_backtest(
        df,
        L_grid=L_grid,
        S_grid=S_grid,
        max_segments=args.max_segments,
    )

    oos_path = out_dir / "oos_equity.csv"
    param_path = out_dir / "rolling_parameters.csv"
    out["oos_equity"].to_csv(oos_path, index=False)
    out["rolling_parameters"].to_csv(param_path, index=False)

    print(f"Saved {oos_path} ({len(out['oos_equity'])} OOS rows)")
    print(f"Saved {param_path} ({len(out['rolling_parameters'])} windows)")
    print(
        f"global_return={out['global_return']:.2f}, "
        f"global_max_drawdown={out['global_max_drawdown']:.2f}, "
        f"n_segments={out['n_segments']}, is_bars={out['is_bars']}, oos_bars={out['oos_bars']}, "
        f"bars_per_trading_day={out['bars_per_trading_day']:.2f}"
    )

    fig = plot_rolling_oos(out["oos_equity"])
    if args.save_fig:
        fig_path = out_dir / "rolling_oos_equity_dd.png"
        fig.savefig(fig_path, dpi=150)
        print(f"Saved {fig_path}")
    if not args.no_show:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
