"""
Week 4: complete performance analysis and structured outputs for rolling OOS backtest.

This module reuses week-3 outputs and does NOT alter core trading logic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data.io import Week4Paths, ensure_output_dir, load_week4_inputs
from group5_config import PRIMARY_DATA_FILE, SECONDARY_DATA_FILE
from main import infer_bars_per_trading_day
from optimization.parameter_stability import plot_parameter_evolution, summarize_parameter_stability
from rolling.performance import build_oos_pnl, compute_performance_summary, rolling_sharpe_series
from rolling.plots import plot_cumulative_return_curve, plot_rolling_sharpe, plot_trade_return_histogram
from strategy.trade_analysis import build_trade_level_table


def _fill_week4_path_defaults(args: argparse.Namespace) -> None:
    """Resolve None path fields from ``--market`` (primary CO vs secondary BTC)."""
    root = Path(__file__).resolve().parent
    if args.market == "secondary":
        mdata = root / SECONDARY_DATA_FILE
        w3 = root / "results" / "week3_btc"
        w4 = root / "results" / "week4_btc"
    else:
        mdata = root / PRIMARY_DATA_FILE
        w3 = root / "results" / "week3"
        w4 = root / "results" / "week4"
    if args.market_data is None:
        args.market_data = str(mdata)
    if args.oos_equity is None:
        args.oos_equity = str(w3 / "oos_equity.csv")
    if args.rolling_parameters is None:
        args.rolling_parameters = str(w3 / "rolling_parameters.csv")
    if args.oos_trades is None:
        args.oos_trades = str(w3 / "oos_trades.csv")
    if args.out_dir is None:
        args.out_dir = str(w4)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Week 4 analysis/export pipeline (professional quant report outputs).")
    p.add_argument(
        "--market",
        choices=("primary", "secondary"),
        default="primary",
        help="Selects default IO folders: primary→results/week3|week4, secondary→week3_btc|week4_btc.",
    )
    p.add_argument(
        "--market-data",
        type=str,
        default=None,
        help="Market CSV (default from --market).",
    )
    p.add_argument(
        "--oos-equity",
        type=str,
        default=None,
        help="Week-3 stitched OOS equity CSV (default from --market).",
    )
    p.add_argument(
        "--rolling-parameters",
        type=str,
        default=None,
        help="Week-3 rolling parameter selections CSV (default from --market).",
    )
    p.add_argument(
        "--oos-trades",
        type=str,
        default=None,
        help="Week-3 completed OOS trades CSV (default from --market).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: results/week4 or results/week4_btc).",
    )
    p.add_argument(
        "--rolling-sharpe-window",
        type=int,
        default=500,
        help="Rolling window size in bars for Sharpe ratio plot.",
    )
    p.add_argument(
        "--price-column",
        type=str,
        default="Close",
        help="Market column used as entry/exit execution proxy price.",
    )
    return p.parse_args()


def run_pipeline(args: argparse.Namespace) -> None:
    paths = Week4Paths(
        market_data=Path(args.market_data),
        oos_equity=Path(args.oos_equity),
        rolling_parameters=Path(args.rolling_parameters),
        oos_trades=Path(args.oos_trades),
        output_dir=Path(args.out_dir),
    )
    ensure_output_dir(paths.output_dir)

    market, oos_equity, rolling_params, oos_trades = load_week4_inputs(paths)

    # Trade-level table: enrich week-3 completed round-turns with entry/exit prices.
    trades = build_trade_level_table(market, oos_trades, price_column=args.price_column)
    trades_path = paths.output_dir / "trades.csv"
    trades.to_csv(trades_path, index=False)

    # Bars/year is inferred from market sampling frequency (reproducible, no hardcoded annualization).
    bars_per_year = float(infer_bars_per_trading_day(market) * 252.0)
    perf = compute_performance_summary(oos_equity, trades, bars_per_year=bars_per_year)
    perf_path = paths.output_dir / "performance_summary.csv"
    perf.to_csv(perf_path, index=False)

    # Parameter stability analysis (table + evolution plot).
    stability = summarize_parameter_stability(rolling_params)
    stability_path = paths.output_dir / "parameter_stability_summary.csv"
    stability.to_csv(stability_path, index=False)
    plot_parameter_evolution(
        rolling_params,
        output_path=paths.output_dir / "parameter_evolution.png",
    )

    # Additional plots requested in Week 4.
    plot_trade_return_histogram(
        trades,
        output_path=paths.output_dir / "trade_returns_histogram.png",
    )
    plot_cumulative_return_curve(
        oos_equity,
        output_path=paths.output_dir / "cumulative_return_curve.png",
    )

    oos_pnl = build_oos_pnl(oos_equity)
    rs = rolling_sharpe_series(
        oos_pnl,
        window=int(args.rolling_sharpe_window),
        bars_per_year=bars_per_year,
    )
    rs_out = pd.DataFrame({"datetime": rs.index, "rolling_sharpe": rs.values})
    rs_out.to_csv(paths.output_dir / "rolling_sharpe.csv", index=False)
    plot_rolling_sharpe(rs, output_path=paths.output_dir / "rolling_sharpe_ratio.png")

    print(f"Saved {trades_path}")
    print(f"Saved {perf_path}")
    print(f"Saved {stability_path}")
    print(f"Saved {paths.output_dir / 'parameter_evolution.png'}")
    print(f"Saved {paths.output_dir / 'trade_returns_histogram.png'}")
    print(f"Saved {paths.output_dir / 'cumulative_return_curve.png'}")
    print(f"Saved {paths.output_dir / 'rolling_sharpe.csv'}")
    print(f"Saved {paths.output_dir / 'rolling_sharpe_ratio.png'}")
    print(f"bars_per_year_used={bars_per_year:.2f}")


def main() -> None:
    args = parse_args()
    _fill_week4_path_defaults(args)
    run_pipeline(args)


if __name__ == "__main__":
    main()

