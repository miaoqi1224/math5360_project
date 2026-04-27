from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    scipy_stats = None

try:
    from scipy import signal as scipy_signal
except Exception:  # pragma: no cover
    scipy_signal = None

try:
    import xlrd
except Exception:  # pragma: no cover
    xlrd = None

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None


PROJECT_DIR = Path(__file__).resolve().parent
TF_DATA_FILE = PROJECT_DIR / "TF Data.xls"
OUTPUT_DIR = PROJECT_DIR / "outputs"
MARKETS = ["CO", "BTC"]

QUICK_MODE = True
INITIAL_CAPITAL = 100_000.0

# Professor grid from the project handout:
# ChnLen = 500:10:10000, StpPct = 0.005:0.001:0.10
FULL_CHANNEL_GRID = np.arange(500, 10001, 10, dtype=np.int64)
FULL_STOP_GRID = np.round(np.arange(0.005, 0.1001, 0.001), 6)

# Faster grid for day-to-day work. Switch QUICK_MODE to False for the full search.
QUICK_CHANNEL_GRID = np.array([500, 1000, 1500, 2000, 2500], dtype=np.int64)
QUICK_STOP_GRID = np.array([0.005, 0.010, 0.015, 0.020, 0.030], dtype=np.float64)

SIGNIFICANCE_LEVEL = 0.05

# 5-minute bars. This grid covers immediate, intraday, and multi-session scales.
TEST_HORIZONS = [1, 2, 3, 6, 12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144, 10000]
VR_HORIZONS = TEST_HORIZONS
PR_HORIZONS = TEST_HORIZONS


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


def resolve_data_file(ticker: str) -> Path:
    path = PROJECT_DIR / f"{ticker}-5minHLV.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    return path


def fallback_market_config(ticker: str) -> MarketConfig:
    fallbacks = {
        "CO": MarketConfig(
            ticker="CO",
            name="Brent Crude",
            exchange="ICE",
            currency="USD",
            point_value=1000.0,
            tick_size=0.01,
            tick_value=10.0,
            slippage=48.0,
        ),
        "BTC": MarketConfig(
            ticker="BTC",
            name="CME Bitcoin",
            exchange="CME",
            currency="USD",
            point_value=5.0,
            tick_size=1.0,
            tick_value=5.0,
            slippage=25.0,
        ),
    }
    return fallbacks[ticker]


def load_market_config(ticker: str) -> MarketConfig:
    fallback = fallback_market_config(ticker)

    if xlrd is None or not TF_DATA_FILE.exists():
        return fallback

    book = xlrd.open_workbook(str(TF_DATA_FILE))
    sheet = book.sheet_by_name("TF Data")
    for row_idx in range(sheet.nrows):
        if str(sheet.cell_value(row_idx, 1)).strip().upper() == ticker:
            point_value = float(sheet.cell_value(row_idx, 7))
            tick_value = float(sheet.cell_value(row_idx, 8))
            return MarketConfig(
                ticker=ticker,
                name=str(sheet.cell_value(row_idx, 3)),
                exchange=str(sheet.cell_value(row_idx, 4)),
                currency=str(sheet.cell_value(row_idx, 5)),
                point_value=point_value,
                tick_size=tick_value / point_value if point_value else fallback.tick_size,
                tick_value=tick_value,
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


def normal_two_sided_p(z_value: float) -> float:
    if not np.isfinite(z_value):
        return float("nan")
    return float(math.erfc(abs(z_value) / math.sqrt(2.0)))


def t_two_sided_p(t_value: float, df: int) -> float:
    if not np.isfinite(t_value) or df <= 0:
        return float("nan")
    if scipy_stats is not None:
        return float(2.0 * scipy_stats.t.sf(abs(t_value), df))
    return normal_two_sided_p(t_value)


def format_p_value(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "n/a"
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.3f}"


def format_horizon(minutes: float) -> str:
    if minutes < 60:
        return f"{minutes:.0f} min"
    hours = minutes / 60.0
    if hours < 24:
        return f"{hours:.1f} hr"
    return f"{hours:.1f} hr (~{hours / 24.0:.1f} days)"


def direction_from_effect(effect: float, p_value: float, baseline: bool = False) -> str:
    if baseline:
        return "baseline"
    if not np.isfinite(effect) or not np.isfinite(p_value):
        return "insufficient data"
    if p_value >= SIGNIFICANCE_LEVEL:
        return "not significant"
    if effect > 0:
        return "trend-following"
    if effect < 0:
        return "mean-reverting"
    return "not significant"


def robust_vr_delta_lags(one_bar_returns: np.ndarray, max_lag: int) -> np.ndarray:
    if max_lag <= 0:
        return np.array([], dtype=float)

    centered = one_bar_returns - np.mean(one_bar_returns)
    squared = centered**2
    denom = float(np.sum(squared) ** 2)
    if denom <= 0:
        return np.full(max_lag, np.nan, dtype=float)

    if scipy_signal is not None and max_lag > 128:
        autocorr = scipy_signal.correlate(squared, squared, mode="full", method="fft")
        center = len(squared) - 1
        lag_sums = autocorr[center + 1 : center + max_lag + 1]
        return lag_sums / denom

    return np.array(
        [float(np.sum(squared[lag:] * squared[:-lag]) / denom) for lag in range(1, max_lag + 1)],
        dtype=float,
    )


def variance_ratio_details(
    one_bar_returns: np.ndarray,
    q: int,
    robust_delta_lags: np.ndarray | None = None,
) -> dict[str, float | str | int]:
    if q <= 0 or len(one_bar_returns) <= q:
        return {
            "n_observations": len(one_bar_returns),
            "variance_ratio": float("nan"),
            "vr_minus_1": float("nan"),
            "vr_z_homoskedastic": float("nan"),
            "vr_p_homoskedastic": float("nan"),
            "vr_z_heteroskedastic": float("nan"),
            "vr_p_heteroskedastic": float("nan"),
            "vr_interpretation": "insufficient data",
        }

    if q == 1:
        return {
            "n_observations": len(one_bar_returns),
            "variance_ratio": 1.0,
            "vr_minus_1": 0.0,
            "vr_z_homoskedastic": float("nan"),
            "vr_p_homoskedastic": float("nan"),
            "vr_z_heteroskedastic": float("nan"),
            "vr_p_heteroskedastic": float("nan"),
            "vr_interpretation": "baseline",
        }

    var_1 = np.var(one_bar_returns, ddof=1)
    if var_1 <= 0:
        return {
            "n_observations": len(one_bar_returns),
            "variance_ratio": float("nan"),
            "vr_minus_1": float("nan"),
            "vr_z_homoskedastic": float("nan"),
            "vr_p_homoskedastic": float("nan"),
            "vr_z_heteroskedastic": float("nan"),
            "vr_p_heteroskedastic": float("nan"),
            "vr_interpretation": "insufficient data",
        }

    cumulative_returns = np.concatenate(([0.0], np.cumsum(one_bar_returns)))
    q_bar_returns = cumulative_returns[q:] - cumulative_returns[:-q]
    var_q = np.var(q_bar_returns, ddof=1)
    vr = float(var_q / (q * var_1))
    vr_minus_1 = vr - 1.0

    n = len(one_bar_returns)
    phi = 2.0 * (2.0 * q - 1.0) * (q - 1.0) / (3.0 * q * n)
    z_homo = vr_minus_1 / math.sqrt(phi) if phi > 0 else float("nan")
    p_homo = normal_two_sided_p(z_homo)

    theta = 0.0
    if robust_delta_lags is not None and len(robust_delta_lags) >= q - 1:
        lags = np.arange(1, q, dtype=float)
        weights = 2.0 * (q - lags) / q
        theta = float(np.sum((weights**2) * robust_delta_lags[: q - 1]))
    else:
        centered = one_bar_returns - np.mean(one_bar_returns)
        denom = float(np.sum(centered**2) ** 2)
        for lag in range(1, q):
            delta = float(np.sum((centered[lag:] ** 2) * (centered[:-lag] ** 2)) / denom)
            weight = 2.0 * (q - lag) / q
            theta += (weight**2) * delta
    z_hetero = vr_minus_1 / math.sqrt(theta) if theta > 0 else float("nan")
    p_hetero = normal_two_sided_p(z_hetero)

    return {
        "n_observations": n,
        "variance_ratio": vr,
        "vr_minus_1": vr_minus_1,
        "vr_z_homoskedastic": z_homo,
        "vr_p_homoskedastic": p_homo,
        "vr_z_heteroskedastic": z_hetero,
        "vr_p_heteroskedastic": p_hetero,
        "vr_interpretation": direction_from_effect(vr_minus_1, p_hetero),
    }


def run_variance_ratio_scan(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].to_numpy(dtype=float)
    rets = compute_log_returns(close)
    max_lag = max(VR_HORIZONS) - 1
    delta_lags = robust_vr_delta_lags(rets, max_lag)

    rows = []
    for h in VR_HORIZONS:
        details = variance_ratio_details(rets, h, delta_lags)
        rows.append(
            {
                "horizon_bars": h,
                "horizon_minutes": h * 5,
                "horizon_label": format_horizon(h * 5),
                **details,
            }
        )
    return pd.DataFrame(rows)


def push_response_for_horizon(log_close: np.ndarray, h: int) -> dict[str, float | str | int]:
    # Adjacent non-overlapping windows match the lecture's push/response setup.
    pushes = log_close[h::h] - log_close[:-h:h]
    responses = log_close[2 * h :: h] - log_close[h:-h:h]

    n = min(len(pushes), len(responses))
    pushes = pushes[:n]
    responses = responses[:n]
    if n < 2:
        return {
            "n_push_response_pairs": n,
            "beta": float("nan"),
            "beta_t_stat": float("nan"),
            "beta_p_value": float("nan"),
            "correlation": float("nan"),
            "r_squared": float("nan"),
            "signed_response": float("nan"),
            "signed_response_bps": float("nan"),
            "signed_response_t_stat": float("nan"),
            "signed_response_p_value": float("nan"),
            "pr_interpretation": "insufficient data",
        }

    x = pushes.astype(float)
    y = responses.astype(float)
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    ssx = float(np.sum(x_centered**2))
    ssy = float(np.sum(y_centered**2))

    if ssx == 0:
        beta = float("nan")
        beta_t = float("nan")
        beta_p = float("nan")
        correlation = float("nan")
        r_squared = float("nan")
    else:
        cov_xy = float(np.sum(x_centered * y_centered))
        beta = cov_xy / ssx
        intercept = float(np.mean(y) - beta * np.mean(x))
        residuals = y - (intercept + beta * x)
        if n > 2:
            residual_var = float(np.sum(residuals**2) / (n - 2))
            beta_se = math.sqrt(residual_var / ssx) if residual_var >= 0 else float("nan")
            beta_t = beta / beta_se if beta_se > 0 else float("nan")
            beta_p = t_two_sided_p(beta_t, n - 2)
        else:
            beta_t = float("nan")
            beta_p = float("nan")
        correlation = cov_xy / math.sqrt(ssx * ssy) if ssy > 0 else float("nan")
        r_squared = correlation**2 if np.isfinite(correlation) else float("nan")

    signed_response_samples = np.sign(x) * y
    signed_response = float(np.mean(signed_response_samples))
    signed_response_std = float(np.std(signed_response_samples, ddof=1)) if n > 1 else 0.0
    if signed_response_std > 0:
        signed_response_t = signed_response / (signed_response_std / math.sqrt(n))
        signed_response_p = t_two_sided_p(signed_response_t, n - 1)
    else:
        signed_response_t = float("nan")
        signed_response_p = float("nan")

    return {
        "n_push_response_pairs": n,
        "beta": beta,
        "beta_t_stat": beta_t,
        "beta_p_value": beta_p,
        "correlation": correlation,
        "r_squared": r_squared,
        "signed_response": signed_response,
        "signed_response_bps": signed_response * 10_000.0,
        "signed_response_t_stat": signed_response_t,
        "signed_response_p_value": signed_response_p,
        "pr_interpretation": direction_from_effect(beta, beta_p),
    }


def run_push_response_scan(df: pd.DataFrame) -> pd.DataFrame:
    log_close = np.log(df["Close"].to_numpy(dtype=float))

    rows = []
    for h in PR_HORIZONS:
        details = push_response_for_horizon(log_close, h)
        rows.append(
            {
                "horizon_bars": h,
                "horizon_minutes": h * 5,
                "horizon_label": format_horizon(h * 5),
                **details,
            }
        )
    return pd.DataFrame(rows)


def combine_rw_tests(vr_df: pd.DataFrame, pr_df: pd.DataFrame) -> pd.DataFrame:
    merged = vr_df.merge(pr_df, on=["horizon_bars", "horizon_minutes"], how="inner")

    def classify(row: pd.Series) -> str:
        vr_type = row["vr_interpretation"]
        pr_type = row["pr_interpretation"]
        directional = {"trend-following", "mean-reverting"}
        if vr_type == "baseline":
            if pr_type in directional:
                return f"push-response only: {pr_type}"
            return "baseline"
        if vr_type == pr_type and vr_type in directional:
            return vr_type
        if vr_type in directional and pr_type == "not significant":
            return f"weak {vr_type}"
        if pr_type in directional and vr_type == "not significant":
            return f"weak {pr_type}"
        if vr_type == "not significant" and pr_type == "not significant":
            return "no clear inefficiency"
        return "mixed evidence"

    if "horizon_label_x" in merged.columns:
        merged["horizon_label"] = merged["horizon_label_x"]
        merged = merged.drop(columns=["horizon_label_x", "horizon_label_y"])
    merged["joint_interpretation"] = merged.apply(classify, axis=1)
    return merged


def interpretation_rank(label: str) -> float:
    ranks = {
        "trend-following": 2.0,
        "weak trend-following": 1.0,
        "push-response only: trend-following": 1.0,
        "mixed evidence": 0.0,
        "baseline": 0.0,
        "no clear inefficiency": 0.0,
        "push-response only: mean-reverting": -1.0,
        "weak mean-reverting": -1.0,
        "mean-reverting": -2.0,
    }
    return ranks.get(label, 0.0)


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


def build_statistical_testing_table(combined_df: pd.DataFrame) -> pd.DataFrame:
    table = combined_df.sort_values("horizon_minutes").copy()
    return pd.DataFrame(
        {
            "Horizon": table["horizon_label"],
            "VR": table["variance_ratio"].map(lambda x: f"{x:.4f}" if np.isfinite(x) else "n/a"),
            "VR robust p": table["vr_p_heteroskedastic"].map(format_p_value),
            "VR signal": table["vr_interpretation"],
            "PR beta": table["beta"].map(lambda x: f"{x:.4f}" if np.isfinite(x) else "n/a"),
            "PR beta p": table["beta_p_value"].map(format_p_value),
            "Signed response (bp)": table["signed_response_bps"].map(
                lambda x: f"{x:.3f}" if np.isfinite(x) else "n/a"
            ),
            "PR signal": table["pr_interpretation"],
            "Joint reading": table["joint_interpretation"],
        }
    )


def save_outputs(
    market: MarketConfig,
    df: pd.DataFrame,
    vr_df: pd.DataFrame,
    pr_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    quarter_df: pd.DataFrame,
    oos_df: pd.DataFrame,
    oos_summary: dict[str, float],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    testing_table = build_statistical_testing_table(combined_df)

    vr_df.to_csv(output_dir / "variance_ratio.csv", index=False)
    pr_df.to_csv(output_dir / "push_response.csv", index=False)
    combined_df.to_csv(output_dir / "random_walk_tests_combined.csv", index=False)
    testing_table.to_csv(output_dir / "statistical_testing_table.csv", index=False)
    quarter_df.to_csv(output_dir / "walk_forward_quarterly_parameters.csv", index=False)
    oos_df.to_csv(output_dir / "walk_forward_oos_equity.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(vr_df["horizon_minutes"], vr_df["variance_ratio"], marker="o")
    significant_vr = vr_df["vr_p_heteroskedastic"] < SIGNIFICANCE_LEVEL
    ax.scatter(
        vr_df.loc[significant_vr, "horizon_minutes"],
        vr_df.loc[significant_vr, "variance_ratio"],
        s=80,
        facecolors="none",
        edgecolors="black",
        label="robust p < 0.05",
    )
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"{market.ticker} Variance Ratio by Horizon")
    ax.set_xlabel("Horizon (minutes)")
    ax.set_ylabel("Variance Ratio")
    ax.set_xscale("log")
    ax.set_xticks(vr_df["horizon_minutes"])
    ax.set_xticklabels(vr_df["horizon_label"], rotation=35, ha="right")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "variance_ratio.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pr_df["horizon_minutes"], pr_df["beta"], marker="o", label="Beta")
    significant_pr = pr_df["beta_p_value"] < SIGNIFICANCE_LEVEL
    ax.scatter(
        pr_df.loc[significant_pr, "horizon_minutes"],
        pr_df.loc[significant_pr, "beta"],
        s=80,
        facecolors="none",
        edgecolors="black",
        label="beta p < 0.05",
    )
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"{market.ticker} Push-Response Beta by Horizon")
    ax.set_xlabel("Horizon (minutes)")
    ax.set_ylabel("Push-Response Beta")
    ax.set_xscale("log")
    ax.set_xticks(pr_df["horizon_minutes"])
    ax.set_xticklabels(pr_df["horizon_label"], rotation=35, ha="right")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "push_response_beta.png", dpi=160)
    plt.close(fig)

    joint_scores = combined_df["joint_interpretation"].map(interpretation_rank)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.scatter(combined_df["horizon_minutes"], joint_scores, s=90)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"{market.ticker} Inefficiency Direction by Time Scale")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Interpretation")
    ax.set_xscale("log")
    ax.set_xticks(combined_df["horizon_minutes"])
    ax.set_xticklabels(combined_df["horizon_label"], rotation=35, ha="right")
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_yticklabels(
        [
            "mean-reverting",
            "weak mean-reverting",
            "mixed/no clear",
            "weak trend-following",
            "trend-following",
        ]
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "inefficiency_timescale.png", dpi=160)
    plt.close(fig)

    if not oos_df.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(oos_df["DateTime"], oos_df["equity"], linewidth=1.2)
        ax.set_title(f"{market.ticker} Walk-Forward Out-of-Sample Equity Curve")
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "walk_forward_oos_equity.png", dpi=160)
        plt.close(fig)


def print_console_summary(
    market: MarketConfig,
    df: pd.DataFrame,
    combined_df: pd.DataFrame,
    quarter_df: pd.DataFrame,
    oos_summary: dict[str, float],
    output_dir: Path,
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
                "horizon_label",
                "variance_ratio",
                "vr_p_heteroskedastic",
                "vr_interpretation",
                "beta",
                "beta_p_value",
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
    print("OOS summary:")
    for key, value in oos_summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    print()
    print(f"Outputs written to: {output_dir}")
    print(f"Grid mode: {'QUICK' if QUICK_MODE else 'FULL'}")


def run_market(ticker: str) -> None:
    market = load_market_config(ticker)
    df = load_data(resolve_data_file(ticker))

    vr_df = run_variance_ratio_scan(df)
    pr_df = run_push_response_scan(df)
    combined_df = combine_rw_tests(vr_df, pr_df)

    quarter_df, oos_df, oos_summary = run_walk_forward(df, market)
    output_dir = OUTPUT_DIR / ticker
    save_outputs(market, df, vr_df, pr_df, combined_df, quarter_df, oos_df, oos_summary, output_dir)
    print_console_summary(market, df, combined_df, quarter_df, oos_summary, output_dir)


def main() -> None:
    tickers = [arg.upper() for arg in sys.argv[1:]] or MARKETS
    for idx, ticker in enumerate(tickers):
        if idx:
            print()
        run_market(ticker)


if __name__ == "__main__":
    main()
