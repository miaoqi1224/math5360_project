"""
Sweep rolling walk-forward over multiple in-sample lengths (T, years) and
out-of-sample lengths (τ, months).

Uses ``main.rolling_backtest`` unchanged. Default grid is modest; use
``--t-years`` / ``--tau-months`` comma lists to customize.

Why vary windows (short answer for write-ups):
- Robustness: conclusions should not hinge on one arbitrary (T, τ) split.
- Bias/variance trade-off: shorter IS fits recent regimes but estimates noisier;
  longer IS stabilizes estimates but may mix stale regimes. Shorter OOS has
  fewer trades per segment; longer OOS yields fewer rolling segments.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from group5_config import PRIMARY_DATA_FILE, SECONDARY_DATA_FILE, contract_slippage_point_value
from main import load_market_data, rolling_backtest


def _fast_grids() -> tuple[np.ndarray, np.ndarray]:
    """Match week3_rolling_oos.py pilot grid."""
    L_grid = np.arange(500, 3001, 500, dtype=int)
    S_grid = np.arange(0.01, 0.0301, 0.01, dtype=float)
    return L_grid, S_grid


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _summarize_one(
    out: dict[str, object],
) -> dict[str, float | int | str]:
    rp: pd.DataFrame = out["rolling_parameters"]
    g_ret = float(out["global_return"])
    g_dd = float(out["global_max_drawdown"])
    rtdd = float(g_ret / abs(g_dd)) if g_dd != 0.0 else float("nan")
    if rp.empty:
        mean_rtdd = med_rtdd = std_rtdd = float("nan")
        nu_l = nu_s = 0
    else:
        col = rp["oos_return_to_dd"].astype(float)
        mean_rtdd = float(col.mean())
        med_rtdd = float(col.median())
        std_rtdd = float(col.std(ddof=1)) if len(col) > 1 else 0.0
        nu_l = int(rp["best_L"].nunique())
        nu_s = int(rp["best_S"].nunique())
    return {
        "n_segments": int(out["n_segments"]),
        "is_bars": int(out["is_bars"]),
        "oos_bars": int(out["oos_bars"]),
        "global_return": g_ret,
        "global_max_drawdown": g_dd,
        "global_return_to_dd": rtdd,
        "mean_oos_return_to_dd": mean_rtdd,
        "median_oos_return_to_dd": med_rtdd,
        "std_oos_return_to_dd": std_rtdd,
        "n_unique_best_L": nu_l,
        "n_unique_best_S": nu_s,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Sensitivity sweep: rolling OOS over multiple (IS years T, OOS months τ)."
    )
    p.add_argument("--market", choices=("primary", "secondary"), default="primary")
    p.add_argument("--data", type=str, default=None)
    p.add_argument(
        "--t-years",
        type=str,
        default="3,4,5",
        help="Comma-separated in-sample lengths in years (T), e.g. 3,4,5",
    )
    p.add_argument(
        "--tau-months",
        type=str,
        default="2,3,6",
        help="Comma-separated OOS horizon in months (τ), e.g. 2,3,6",
    )
    p.add_argument("--fast-grid", action="store_true", help="Use small L/S grid (recommended).")
    p.add_argument(
        "--max-segments",
        type=int,
        default=None,
        help="Cap rolling windows per combo (debug only).",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path (default under results/sensitivity/).",
    )
    args = p.parse_args()

    t_list = _parse_float_list(args.t_years)
    tau_list = _parse_float_list(args.tau_months)

    data_path = args.data or (SECONDARY_DATA_FILE if args.market == "secondary" else PRIMARY_DATA_FILE)
    tag = "fastgrid" if args.fast_grid else "fullgrid"
    mlabel = "btc" if args.market == "secondary" else "co"
    out_path = Path(
        args.out or f"results/sensitivity/window_sweep_{mlabel}_{tag}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_market_data(data_path)
    if args.fast_grid:
        L_grid, S_grid = _fast_grids()
    else:
        L_grid, S_grid = None, None

    slpg_f, pv_f = contract_slippage_point_value(args.market)

    rows: list[dict[str, object]] = []
    for T in t_list:
        for tau in tau_list:
            try:
                out = rolling_backtest(
                    df,
                    is_years=float(T),
                    oos_months=float(tau),
                    L_grid=L_grid,
                    S_grid=S_grid,
                    max_segments=args.max_segments,
                    slpg=slpg_f,
                    pv=pv_f,
                )
            except ValueError as e:
                print(f"SKIP T={T} τ={tau} months: {e}")
                continue
            s = _summarize_one(out)
            rows.append(
                {
                    "market": args.market,
                    "data_file": str(data_path),
                    "is_years": float(T),
                    "oos_months": float(tau),
                    **s,
                }
            )
            print(
                f"OK T={T} τ={tau}m -> n_seg={s['n_segments']}, "
                f"g_ret={s['global_return']:.2f}, g_dd={s['global_max_drawdown']:.2f}, "
                f"mean_oos_rtdd={s['mean_oos_return_to_dd']:.4f}"
            )

    if not rows:
        raise SystemExit("No successful runs; check T/τ and data length.")

    res = pd.DataFrame(rows).sort_values(
        ["is_years", "oos_months"], ascending=[True, True]
    )
    res.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(res)} rows)")


if __name__ == "__main__":
    main()
