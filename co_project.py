from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import xlrd
except Exception:  # pragma: no cover
    xlrd = None

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None


PROJECT_DIR = Path("/Users/regina/Desktop/5360 project")
DATA_FILE = PROJECT_DIR / "CO-5minHLV.csv"
TF_DATA_FILE = PROJECT_DIR / "TF Data.xls"
OUTPUT_DIR = PROJECT_DIR / "outputs"

QUICK_MODE = True
INITIAL_CAPITAL = 100_000.0

# Professor grid from the project handout:
# ChnLen = 500:10:10000, StpPct = 0.005:0.001:0.10
FULL_CHANNEL_GRID = np.arange(500, 10001, 10, dtype=np.int64)
FULL_STOP_GRID = np.round(np.arange(0.005, 0.1001, 0.001), 6)

# Faster grid for day-to-day work. Switch QUICK_MODE to False for the full search.
QUICK_CHANNEL_GRID = np.array([500, 1000, 1500, 2000, 2500], dtype=np.int64)
QUICK_STOP_GRID = np.array([0.005, 0.010, 0.015, 0.020, 0.030], dtype=np.float64)

VR_HORIZONS = [1, 3, 6, 12, 24, 48, 96]
PR_HORIZONS = [1, 3, 6, 12, 24, 48, 96]


@dataclass
class MarketConfig:
    ticker: str
    name: str
    exchange: str
    currency: str
    point_value: float
    tick_size: float
    tick_value: float
    slippage: float


def load_market_config() -> MarketConfig:
    fallback = MarketConfig(
        ticker="CO",
        name="Brent Crude",
        exchange="ICE",
        currency="USD",
        point_value=1000.0,
        tick_size=0.01,
        tick_value=10.0,
        slippage=48.0,
    )

    if xlrd is None or not TF_DATA_FILE.exists():
        return fallback

    book = xlrd.open_workbook(str(TF_DATA_FILE))
    sheet = book.sheet_by_name("TF Data")
    for row_idx in range(sheet.nrows):
        if sheet.cell_value(row_idx, 1) == "CO":
            return MarketConfig(
                ticker="CO",
                name=str(sheet.cell_value(row_idx, 3)),
                exchange=str(sheet.cell_value(row_idx, 4)),
                currency=str(sheet.cell_value(row_idx, 5)),
                point_value=float(sheet.cell_value(row_idx, 7)),
                tick_size=0.01,
                tick_value=float(sheet.cell_value(row_idx, 8)),
                slippage=float(sheet.cell_value(row_idx, 21)),
            )

    return fallback


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%m/%d/%Y %H:%M")
    df = df.sort_values("DateTime").reset_index(drop=True)
    return df


def compute_log_returns(close: np.ndarray) -> np.ndarray:
    return np.diff(np.log(close))


def variance_ratio(one_bar_returns: np.ndarray, q: int) -> float:
    if q <= 0 or len(one_bar_returns) <= q:
        return float("nan")

    var_1 = np.var(one_bar_returns, ddof=1)
    if var_1 == 0:
        return float("nan")

    agg = np.convolve(one_bar_returns, np.ones(q), mode="valid")
    var_q = np.var(agg, ddof=1)
    return float(var_q / (q * var_1))


def interpret_vr(vr: float) -> str:
    if np.isnan(vr):
        return "insufficient data"
    if vr > 1.02:
        return "trend-following"
    if vr < 0.98:
        return "mean-reverting"
    return "close to random walk"


def run_variance_ratio_scan(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].to_numpy(dtype=float)
    rets = compute_log_returns(close)

    rows = []
    for h in VR_HORIZONS:
        vr = variance_ratio(rets, h)
        rows.append(
            {
                "horizon_bars": h,
                "horizon_minutes": h * 5,
                "variance_ratio": vr,
                "vr_interpretation": interpret_vr(vr),
            }
        )
    return pd.DataFrame(rows)


def push_response_for_horizon(close: np.ndarray, h: int) -> tuple[float, float]:
    pushes = close[h::h] - close[:-h:h]
    responses = close[2 * h :: h] - close[h:-h:h]

    n = min(len(pushes), len(responses))
    pushes = pushes[:n]
    responses = responses[:n]
    if n < 2:
        return float("nan"), float("nan")

    var_push = np.var(pushes, ddof=1)
    if var_push == 0:
        beta = float("nan")
    else:
        beta = float(np.cov(pushes, responses, ddof=1)[0, 1] / var_push)

    signed_resp = float(np.mean(np.sign(pushes) * responses))
    return beta, signed_resp


def interpret_push_response(beta: float) -> str:
    if np.isnan(beta):
        return "insufficient data"
    if beta > 0:
        return "trend-following"
    if beta < 0:
        return "mean-reverting"
    return "close to random walk"


def run_push_response_scan(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].to_numpy(dtype=float)

    rows = []
    for h in PR_HORIZONS:
        beta, signed_resp = push_response_for_horizon(close, h)
        rows.append(
            {
                "horizon_bars": h,
                "horizon_minutes": h * 5,
                "beta": beta,
                "signed_response": signed_resp,
                "pr_interpretation": interpret_push_response(beta),
            }
        )
    return pd.DataFrame(rows)


def combine_rw_tests(vr_df: pd.DataFrame, pr_df: pd.DataFrame) -> pd.DataFrame:
    merged = vr_df.merge(pr_df, on=["horizon_bars", "horizon_minutes"], how="inner")

    def classify(row: pd.Series) -> str:
        vr_type = row["vr_interpretation"]
        pr_type = row["pr_interpretation"]
        if vr_type == pr_type and vr_type in {"trend-following", "mean-reverting"}:
            return vr_type
        if vr_type == "close to random walk" and pr_type == "close to random walk":
            return "close to random walk"
        return "mixed evidence"

    merged["joint_interpretation"] = merged.apply(classify, axis=1)
    return merged


def rolling_channel(high: np.ndarray, low: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    hh = high_s.rolling(length).max().shift(1).to_numpy(dtype=np.float64)
    ll = low_s.rolling(length).min().shift(1).to_numpy(dtype=np.float64)
    return hh, ll


if njit is not None:

    @njit(cache=True)
    def backtest_path_numba(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        hh: np.ndarray,
        ll: np.ndarray,
        start_idx: int,
        end_idx: int,
        stop_pct: float,
        point_value: float,
        slippage: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        pnl = np.zeros(end_idx - start_idx, dtype=np.float64)
        trades = np.zeros(end_idx - start_idx, dtype=np.float64)

        position = 0
        benchmark_long = 0.0
        benchmark_short = 0.0

        for k in range(start_idx, end_idx):
            local_idx = k - start_idx
            if math.isnan(hh[k]) or math.isnan(ll[k]):
                continue

            traded = False
            delta = 0.0
            if k > 0:
                delta = point_value * (close[k] - close[k - 1]) * position

            if position == 0:
                buy = high[k] >= hh[k]
                sell = low[k] <= ll[k]

                if buy and sell:
                    delta = -slippage + point_value * (ll[k] - hh[k])
                    trades[local_idx] = 1.0
                else:
                    if buy:
                        delta = -slippage / 2.0 + point_value * (close[k] - hh[k])
                        position = 1
                        traded = True
                        benchmark_long = high[k]
                        trades[local_idx] = 0.5
                    if sell:
                        delta = -slippage / 2.0 - point_value * (close[k] - ll[k])
                        position = -1
                        traded = True
                        benchmark_short = low[k]
                        trades[local_idx] = 0.5

            if position == 1 and not traded:
                sell_short = low[k] <= ll[k]
                sell = low[k] <= (benchmark_long * (1.0 - stop_pct))

                if sell_short and sell:
                    delta = delta - slippage - 2.0 * point_value * (close[k] - ll[k])
                    position = -1
                    benchmark_short = low[k]
                    trades[local_idx] = 1.0
                else:
                    if sell:
                        delta = (
                            delta
                            - slippage / 2.0
                            - point_value * (close[k] - benchmark_long * (1.0 - stop_pct))
                        )
                        position = 0
                        trades[local_idx] = 0.5

                    if sell_short:
                        delta = delta - slippage - 2.0 * point_value * (close[k] - ll[k])
                        position = -1
                        benchmark_short = low[k]
                        trades[local_idx] = 1.0

                if high[k] > benchmark_long:
                    benchmark_long = high[k]

            if position == -1 and not traded:
                buy_long = high[k] >= hh[k]
                buy = high[k] >= (benchmark_short * (1.0 + stop_pct))

                if buy_long and buy:
                    delta = delta - slippage + 2.0 * point_value * (close[k] - hh[k])
                    position = 1
                    benchmark_long = high[k]
                    trades[local_idx] = 1.0
                else:
                    if buy:
                        delta = (
                            delta
                            - slippage / 2.0
                            + point_value * (close[k] - benchmark_short * (1.0 + stop_pct))
                        )
                        position = 0
                        trades[local_idx] = 0.5

                    if buy_long:
                        delta = delta - slippage + 2.0 * point_value * (close[k] - hh[k])
                        position = 1
                        benchmark_long = high[k]
                        trades[local_idx] = 1.0

                if low[k] < benchmark_short:
                    benchmark_short = low[k]

            pnl[local_idx] = delta

        return pnl, trades

else:

    def backtest_path_numba(*args, **kwargs):  # pragma: no cover
        raise RuntimeError("numba is required for backtest_path_numba")


def compute_segment_stats(pnl: np.ndarray, trades: np.ndarray) -> dict[str, float]:
    if len(pnl) == 0:
        return {
            "net_profit": 0.0,
            "worst_drawdown": 0.0,
            "pnl_std": 0.0,
            "trade_count": 0.0,
            "return_over_drawdown": np.nan,
            "sharpe_like": np.nan,
        }

    equity = np.cumsum(pnl) + INITIAL_CAPITAL
    running_max = np.maximum.accumulate(equity)
    drawdown = equity - running_max

    net_profit = float(np.sum(pnl))
    worst_drawdown = float(np.min(drawdown))
    pnl_std = float(np.std(pnl, ddof=1)) if len(pnl) > 1 else 0.0
    trade_count = float(np.sum(trades))
    if worst_drawdown < 0:
        rod = net_profit / abs(worst_drawdown)
    elif net_profit > 0:
        rod = float("inf")
    else:
        rod = np.nan
    sharpe_like = float(np.mean(pnl) / pnl_std) if pnl_std > 0 else np.nan

    return {
        "net_profit": net_profit,
        "worst_drawdown": worst_drawdown,
        "pnl_std": pnl_std,
        "trade_count": trade_count,
        "return_over_drawdown": rod,
        "sharpe_like": sharpe_like,
    }


def build_walk_forward_windows(dt: pd.Series) -> list[dict[str, object]]:
    first_dt = dt.iloc[0]
    last_dt = dt.iloc[-1]
    oos_start = first_dt + pd.DateOffset(years=4)
    quarter = pd.DateOffset(months=3)

    windows: list[dict[str, object]] = []
    while oos_start + quarter <= last_dt:
        is_start = oos_start - pd.DateOffset(years=4)
        is_end = oos_start
        oos_end = oos_start + quarter

        is_start_idx = int(dt.searchsorted(is_start))
        is_end_idx = int(dt.searchsorted(is_end))
        oos_end_idx = int(dt.searchsorted(oos_end))

        windows.append(
            {
                "is_start": is_start,
                "is_end": is_end,
                "oos_start": oos_start,
                "oos_end": oos_end,
                "is_start_idx": is_start_idx,
                "is_end_idx": is_end_idx,
                "oos_end_idx": oos_end_idx,
            }
        )
        oos_start = oos_end

    return windows


def evaluate_single_window(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    dt: pd.Series,
    window: dict[str, object],
    channel_grid: np.ndarray,
    stop_grid: np.ndarray,
    market: MarketConfig,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    is_start_idx = int(window["is_start_idx"])
    is_end_idx = int(window["is_end_idx"])
    oos_end_idx = int(window["oos_end_idx"])

    best: dict[str, float | int | str] | None = None
    grid_rows: list[dict[str, float]] = []

    for length in channel_grid:
        hh, ll = rolling_channel(high, low, int(length))
        sim_start_idx = max(1, is_start_idx - int(length))
        pnl_all_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}

        for stop_pct in stop_grid:
            pnl_all, trades_all = backtest_path_numba(
                high,
                low,
                close,
                hh,
                ll,
                sim_start_idx,
                oos_end_idx,
                float(stop_pct),
                market.point_value,
                market.slippage,
            )
            pnl_all_cache[float(stop_pct)] = (pnl_all, trades_all)

            is_local_start = is_start_idx - sim_start_idx
            is_local_end = is_end_idx - sim_start_idx
            is_stats = compute_segment_stats(
                pnl_all[is_local_start:is_local_end],
                trades_all[is_local_start:is_local_end],
            )

            row = {
                "channel_length": int(length),
                "stop_pct": float(stop_pct),
                "is_net_profit": is_stats["net_profit"],
                "is_worst_drawdown": is_stats["worst_drawdown"],
                "is_trade_count": is_stats["trade_count"],
                "is_return_over_drawdown": is_stats["return_over_drawdown"],
            }
            grid_rows.append(row)

            objective = is_stats["return_over_drawdown"]
            objective_rank = objective if np.isfinite(objective) else is_stats["net_profit"]

            best_rank = None if best is None else float(best["objective_rank"])
            if best is None or objective_rank > best_rank:
                best = {
                    **row,
                    "sim_start_idx": sim_start_idx,
                    "objective_rank": objective_rank,
                }

    if best is None:
        raise RuntimeError("No valid parameter combination found in window optimization.")

    best_length = int(best["channel_length"])
    best_stop = float(best["stop_pct"])
    hh, ll = rolling_channel(high, low, best_length)
    sim_start_idx = int(best["sim_start_idx"])
    pnl_all, trades_all = backtest_path_numba(
        high,
        low,
        close,
        hh,
        ll,
        sim_start_idx,
        oos_end_idx,
        best_stop,
        market.point_value,
        market.slippage,
    )

    oos_local_start = is_end_idx - sim_start_idx
    oos_local_end = oos_end_idx - sim_start_idx
    oos_stats = compute_segment_stats(
        pnl_all[oos_local_start:oos_local_end],
        trades_all[oos_local_start:oos_local_end],
    )

    detail = pd.DataFrame(
        {
            "DateTime": dt.iloc[is_end_idx:oos_end_idx].to_numpy(),
            "oos_pnl": pnl_all[oos_local_start:oos_local_end],
            "oos_trades": trades_all[oos_local_start:oos_local_end],
        }
    )

    result = {
        "is_start": pd.Timestamp(window["is_start"]),
        "is_end": pd.Timestamp(window["is_end"]),
        "oos_start": pd.Timestamp(window["oos_start"]),
        "oos_end": pd.Timestamp(window["oos_end"]),
        "channel_length": best_length,
        "stop_pct": best_stop,
        "is_return_over_drawdown": float(best["is_return_over_drawdown"]),
        "is_net_profit": float(best["is_net_profit"]),
        "is_worst_drawdown": float(best["is_worst_drawdown"]),
        "oos_net_profit": oos_stats["net_profit"],
        "oos_worst_drawdown": oos_stats["worst_drawdown"],
        "oos_trade_count": oos_stats["trade_count"],
        "oos_return_over_drawdown": oos_stats["return_over_drawdown"],
        "oos_sharpe_like": oos_stats["sharpe_like"],
    }
    return result, detail


def run_walk_forward(df: pd.DataFrame, market: MarketConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    channel_grid = QUICK_CHANNEL_GRID if QUICK_MODE else FULL_CHANNEL_GRID
    stop_grid = QUICK_STOP_GRID if QUICK_MODE else FULL_STOP_GRID

    high = df["High"].to_numpy(dtype=np.float64)
    low = df["Low"].to_numpy(dtype=np.float64)
    close = df["Close"].to_numpy(dtype=np.float64)

    windows = build_walk_forward_windows(df["DateTime"])
    quarter_results: list[dict[str, float | int | str]] = []
    oos_parts: list[pd.DataFrame] = []

    for idx, window in enumerate(windows, start=1):
        result, detail = evaluate_single_window(
            high, low, close, df["DateTime"], window, channel_grid, stop_grid, market
        )
        result["window_number"] = idx
        quarter_results.append(result)
        detail["window_number"] = idx
        oos_parts.append(detail)

    quarter_df = pd.DataFrame(quarter_results)
    oos_df = pd.concat(oos_parts, ignore_index=True) if oos_parts else pd.DataFrame(columns=["DateTime", "oos_pnl", "oos_trades", "window_number"])

    if not oos_df.empty:
        oos_df["equity"] = INITIAL_CAPITAL + oos_df["oos_pnl"].cumsum()
    else:
        oos_df["equity"] = []

    summary = compute_segment_stats(
        oos_df["oos_pnl"].to_numpy(dtype=float) if not oos_df.empty else np.array([]),
        oos_df["oos_trades"].to_numpy(dtype=float) if not oos_df.empty else np.array([]),
    )
    return quarter_df, oos_df, summary


def save_outputs(
    market: MarketConfig,
    df: pd.DataFrame,
    vr_df: pd.DataFrame,
    pr_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    quarter_df: pd.DataFrame,
    oos_df: pd.DataFrame,
    oos_summary: dict[str, float],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    vr_df.to_csv(OUTPUT_DIR / "variance_ratio.csv", index=False)
    pr_df.to_csv(OUTPUT_DIR / "push_response.csv", index=False)
    combined_df.to_csv(OUTPUT_DIR / "random_walk_tests_combined.csv", index=False)
    quarter_df.to_csv(OUTPUT_DIR / "walk_forward_quarterly_parameters.csv", index=False)
    oos_df.to_csv(OUTPUT_DIR / "walk_forward_oos_equity.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(vr_df["horizon_minutes"], vr_df["variance_ratio"], marker="o")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("CO Variance Ratio by Horizon")
    ax.set_xlabel("Horizon (minutes)")
    ax.set_ylabel("Variance Ratio")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "variance_ratio.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pr_df["horizon_minutes"], pr_df["beta"], marker="o", label="Beta")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("CO Push-Response Beta by Horizon")
    ax.set_xlabel("Horizon (minutes)")
    ax.set_ylabel("Push-Response Beta")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "push_response_beta.png", dpi=160)
    plt.close(fig)

    if not oos_df.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(oos_df["DateTime"], oos_df["equity"], linewidth=1.2)
        ax.set_title("CO Walk-Forward Out-of-Sample Equity Curve")
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "walk_forward_oos_equity.png", dpi=160)
        plt.close(fig)

    with open(OUTPUT_DIR / "summary.md", "w", encoding="utf-8") as f:
        f.write(f"# {market.ticker} {market.name} Final Project Summary\n\n")
        f.write("## Market\n")
        f.write(f"- Exchange: {market.exchange}\n")
        f.write(f"- Currency: {market.currency}\n")
        f.write(f"- Point value: {market.point_value}\n")
        f.write(f"- Tick size: {market.tick_size}\n")
        f.write(f"- Tick value: {market.tick_value}\n")
        f.write(f"- Slippage: {market.slippage}\n")
        f.write(f"- Sample bars: {len(df)}\n")
        f.write(f"- Sample start: {df['DateTime'].iloc[0]}\n")
        f.write(f"- Sample end: {df['DateTime'].iloc[-1]}\n\n")
        f.write("## Random Walk Tests\n")
        f.write("```text\n")
        f.write(combined_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        f.write("\n```\n\n")
        f.write("## Walk-Forward Out-of-Sample Summary\n")
        for key, value in oos_summary.items():
            f.write(f"- {key}: {value}\n")
        f.write("\n")
        f.write(
            f"Grid mode: {'QUICK' if QUICK_MODE else 'FULL'} "
            f"({len(QUICK_CHANNEL_GRID) if QUICK_MODE else len(FULL_CHANNEL_GRID)} channel values, "
            f"{len(QUICK_STOP_GRID) if QUICK_MODE else len(FULL_STOP_GRID)} stop values)\n"
        )


def print_console_summary(
    market: MarketConfig,
    df: pd.DataFrame,
    combined_df: pd.DataFrame,
    quarter_df: pd.DataFrame,
    oos_summary: dict[str, float],
) -> None:
    print(f"Market: {market.ticker} ({market.name})")
    print(f"Exchange: {market.exchange}")
    print(f"Currency: {market.currency}")
    print(f"Point value: {market.point_value}")
    print(f"Tick value: {market.tick_value}")
    print(f"Slippage: {market.slippage}")
    print(f"Loaded bars: {len(df)}")
    print(f"Sample: {df['DateTime'].iloc[0]} -> {df['DateTime'].iloc[-1]}")
    print()
    print("Random Walk tests:")
    print(
        combined_df[
            [
                "horizon_minutes",
                "variance_ratio",
                "vr_interpretation",
                "beta",
                "pr_interpretation",
                "joint_interpretation",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )
    print()
    print(f"Walk-forward windows: {len(quarter_df)}")
    if not quarter_df.empty:
        print("First five quarterly optimal parameters:")
        print(
            quarter_df[
                [
                    "window_number",
                    "oos_start",
                    "channel_length",
                    "stop_pct",
                    "oos_net_profit",
                    "oos_return_over_drawdown",
                ]
            ]
            .head()
            .to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )
    print()
    print("Overall out-of-sample summary:")
    for key, value in oos_summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    print()
    print(f"Outputs written to: {OUTPUT_DIR}")
    print(f"Grid mode: {'QUICK' if QUICK_MODE else 'FULL'}")


def main() -> None:
    market = load_market_config()
    df = load_data(DATA_FILE)

    vr_df = run_variance_ratio_scan(df)
    pr_df = run_push_response_scan(df)
    combined_df = combine_rw_tests(vr_df, pr_df)

    quarter_df, oos_df, oos_summary = run_walk_forward(df, market)
    save_outputs(market, df, vr_df, pr_df, combined_df, quarter_df, oos_df, oos_summary)
    print_console_summary(market, df, combined_df, quarter_df, oos_summary)


if __name__ == "__main__":
    main()
