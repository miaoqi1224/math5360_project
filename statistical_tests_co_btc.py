from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy import signal as scipy_signal
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    scipy_signal = None
    scipy_stats = None

try:
    import xlrd
except Exception:  # pragma: no cover
    xlrd = None


PROJECT_DIR = Path(__file__).resolve().parent
TF_DATA_FILE = PROJECT_DIR / "TF Data.xls"
OUTPUT_DIR = PROJECT_DIR / "outputs"

MARKETS = ("CO", "BTC")
SIGNIFICANCE_LEVEL = 0.05

# 5-minute bars. Horizons cover short intraday through multi-day scales.
TEST_HORIZONS = (1, 2, 3, 6, 12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144, 10000)


@dataclass(frozen=True)
class MarketConfig:
    ticker: str
    name: str
    exchange: str
    currency: str
    point_value: float
    tick_size: float
    tick_value: float
    slippage: float


FALLBACK_CONFIGS = {
    "CO": MarketConfig("CO", "Brent Crude", "ICE", "USD", 1000.0, 0.01, 10.0, 48.0),
    "BTC": MarketConfig("BTC", "CME Bitcoin", "CME", "USD", 5.0, 1.0, 5.0, 25.0),
}


def data_file(ticker: str) -> Path:
    path = PROJECT_DIR / f"{ticker}-5minHLV.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    return path


def load_market_config(ticker: str) -> MarketConfig:
    fallback = FALLBACK_CONFIGS[ticker]
    if xlrd is None or not TF_DATA_FILE.exists():
        return fallback

    book = xlrd.open_workbook(str(TF_DATA_FILE))
    sheet = book.sheet_by_name("TF Data")
    for row_idx in range(sheet.nrows):
        if str(sheet.cell_value(row_idx, 1)).strip().upper() != ticker:
            continue

        point_value = float(sheet.cell_value(row_idx, 7))
        tick_value = float(sheet.cell_value(row_idx, 8))
        tick_size = tick_value / point_value if point_value else fallback.tick_size
        return MarketConfig(
            ticker=ticker,
            name=str(sheet.cell_value(row_idx, 3)),
            exchange=str(sheet.cell_value(row_idx, 4)),
            currency=str(sheet.cell_value(row_idx, 5)),
            point_value=point_value,
            tick_size=tick_size,
            tick_value=tick_value,
            slippage=float(sheet.cell_value(row_idx, 21)),
        )
    return fallback


def load_price_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["DateTime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%m/%d/%Y %H:%M",
        errors="coerce",
    )
    if df["DateTime"].isna().any():
        df["DateTime"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str))
    return df.sort_values("DateTime").reset_index(drop=True)


def normal_two_sided_p(z_value: float) -> float:
    if not np.isfinite(z_value):
        return float("nan")
    return float(math.erfc(abs(z_value) / math.sqrt(2.0)))


def t_two_sided_p(t_value: float, degrees_of_freedom: int) -> float:
    if not np.isfinite(t_value) or degrees_of_freedom <= 0:
        return float("nan")
    if scipy_stats is not None:
        return float(2.0 * scipy_stats.t.sf(abs(t_value), degrees_of_freedom))
    return normal_two_sided_p(t_value)


def p_label(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "n/a"
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.3f}"


def horizon_label(minutes: float) -> str:
    if minutes < 60:
        return f"{minutes:.0f} min"
    hours = minutes / 60.0
    if hours < 24:
        return f"{hours:.1f} hr"
    return f"{hours:.1f} hr (~{hours / 24.0:.1f} days)"


def signal_label(effect: float, p_value: float, baseline: bool = False) -> str:
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


def log_returns(close: np.ndarray) -> np.ndarray:
    return np.diff(np.log(close))


def robust_vr_delta_lags(returns: np.ndarray, max_lag: int) -> np.ndarray:
    if max_lag <= 0:
        return np.array([], dtype=float)

    centered = returns - np.mean(returns)
    squared = centered**2
    denom = float(np.sum(squared) ** 2)
    if denom <= 0:
        return np.full(max_lag, np.nan, dtype=float)

    if scipy_signal is not None and max_lag > 128:
        autocorr = scipy_signal.correlate(squared, squared, mode="full", method="fft")
        center = len(squared) - 1
        return autocorr[center + 1 : center + max_lag + 1] / denom

    return np.array(
        [float(np.sum(squared[lag:] * squared[:-lag]) / denom) for lag in range(1, max_lag + 1)]
    )


def variance_ratio_row(returns: np.ndarray, q: int, delta_lags: np.ndarray) -> dict[str, float | int | str]:
    if q <= 0 or len(returns) <= q:
        return {
            "n_observations": len(returns),
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
            "n_observations": len(returns),
            "variance_ratio": 1.0,
            "vr_minus_1": 0.0,
            "vr_z_homoskedastic": float("nan"),
            "vr_p_homoskedastic": float("nan"),
            "vr_z_heteroskedastic": float("nan"),
            "vr_p_heteroskedastic": float("nan"),
            "vr_interpretation": "baseline",
        }

    var_1 = np.var(returns, ddof=1)
    if var_1 <= 0:
        return {
            "n_observations": len(returns),
            "variance_ratio": float("nan"),
            "vr_minus_1": float("nan"),
            "vr_z_homoskedastic": float("nan"),
            "vr_p_homoskedastic": float("nan"),
            "vr_z_heteroskedastic": float("nan"),
            "vr_p_heteroskedastic": float("nan"),
            "vr_interpretation": "insufficient data",
        }

    cumulative = np.concatenate(([0.0], np.cumsum(returns)))
    q_returns = cumulative[q:] - cumulative[:-q]
    variance_ratio = float(np.var(q_returns, ddof=1) / (q * var_1))
    vr_minus_1 = variance_ratio - 1.0

    n = len(returns)
    phi = 2.0 * (2.0 * q - 1.0) * (q - 1.0) / (3.0 * q * n)
    z_homo = vr_minus_1 / math.sqrt(phi) if phi > 0 else float("nan")
    p_homo = normal_two_sided_p(z_homo)

    lags = np.arange(1, q, dtype=float)
    weights = 2.0 * (q - lags) / q
    theta = float(np.sum((weights**2) * delta_lags[: q - 1]))
    z_hetero = vr_minus_1 / math.sqrt(theta) if theta > 0 else float("nan")
    p_hetero = normal_two_sided_p(z_hetero)

    return {
        "n_observations": n,
        "variance_ratio": variance_ratio,
        "vr_minus_1": vr_minus_1,
        "vr_z_homoskedastic": z_homo,
        "vr_p_homoskedastic": p_homo,
        "vr_z_heteroskedastic": z_hetero,
        "vr_p_heteroskedastic": p_hetero,
        "vr_interpretation": signal_label(vr_minus_1, p_hetero),
    }


def run_variance_ratio(df: pd.DataFrame) -> pd.DataFrame:
    returns = log_returns(df["Close"].to_numpy(dtype=float))
    delta_lags = robust_vr_delta_lags(returns, max(TEST_HORIZONS) - 1)
    rows = []
    for q in TEST_HORIZONS:
        rows.append(
            {
                "horizon_bars": q,
                "horizon_minutes": q * 5,
                "horizon_label": horizon_label(q * 5),
                **variance_ratio_row(returns, q, delta_lags),
            }
        )
    return pd.DataFrame(rows)


def push_response_row(log_close: np.ndarray, q: int) -> dict[str, float | int | str]:
    pushes = log_close[q::q] - log_close[:-q:q]
    responses = log_close[2 * q :: q] - log_close[q:-q:q]

    n = min(len(pushes), len(responses))
    x = pushes[:n].astype(float)
    y = responses[:n].astype(float)
    if n < 3:
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

    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    ssx = float(np.sum(x_centered**2))
    ssy = float(np.sum(y_centered**2))

    if ssx > 0:
        cov_xy = float(np.sum(x_centered * y_centered))
        beta = cov_xy / ssx
        intercept = float(np.mean(y) - beta * np.mean(x))
        residuals = y - (intercept + beta * x)
        residual_var = float(np.sum(residuals**2) / (n - 2))
        beta_se = math.sqrt(residual_var / ssx) if residual_var >= 0 else float("nan")
        beta_t = beta / beta_se if beta_se > 0 else float("nan")
        beta_p = t_two_sided_p(beta_t, n - 2)
        corr = cov_xy / math.sqrt(ssx * ssy) if ssy > 0 else float("nan")
        r_squared = corr**2 if np.isfinite(corr) else float("nan")
    else:
        beta = beta_t = beta_p = corr = r_squared = float("nan")

    signed_samples = np.sign(x) * y
    signed_response = float(np.mean(signed_samples))
    signed_std = float(np.std(signed_samples, ddof=1))
    signed_t = signed_response / (signed_std / math.sqrt(n)) if signed_std > 0 else float("nan")
    signed_p = t_two_sided_p(signed_t, n - 1)

    return {
        "n_push_response_pairs": n,
        "beta": beta,
        "beta_t_stat": beta_t,
        "beta_p_value": beta_p,
        "correlation": corr,
        "r_squared": r_squared,
        "signed_response": signed_response,
        "signed_response_bps": signed_response * 10000.0,
        "signed_response_t_stat": signed_t,
        "signed_response_p_value": signed_p,
        "pr_interpretation": signal_label(beta, beta_p),
    }


def run_push_response(df: pd.DataFrame) -> pd.DataFrame:
    log_close = np.log(df["Close"].to_numpy(dtype=float))
    rows = []
    for q in TEST_HORIZONS:
        rows.append(
            {
                "horizon_bars": q,
                "horizon_minutes": q * 5,
                "horizon_label": horizon_label(q * 5),
                **push_response_row(log_close, q),
            }
        )
    return pd.DataFrame(rows)


def combine_tests(vr_df: pd.DataFrame, pr_df: pd.DataFrame) -> pd.DataFrame:
    merged = vr_df.merge(pr_df, on=("horizon_bars", "horizon_minutes"), suffixes=("_vr", "_pr"))
    merged["horizon_label"] = merged["horizon_label_vr"]
    merged = merged.drop(columns=["horizon_label_vr", "horizon_label_pr"])

    def classify(row: pd.Series) -> str:
        vr = row["vr_interpretation"]
        pr = row["pr_interpretation"]
        directional = {"trend-following", "mean-reverting"}
        if vr == "baseline":
            return f"push-response only: {pr}" if pr in directional else "baseline"
        if vr == pr and vr in directional:
            return vr
        if vr in directional and pr == "not significant":
            return f"weak {vr}"
        if pr in directional and vr == "not significant":
            return f"weak {pr}"
        if vr == "not significant" and pr == "not significant":
            return "no clear inefficiency"
        return "mixed evidence"

    merged["joint_interpretation"] = merged.apply(classify, axis=1)
    return merged


def interpretation_score(label: str) -> float:
    scores = {
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
    return scores.get(label, 0.0)


def compact_ranges(df: pd.DataFrame, labels: set[str]) -> str:
    ordered = df.sort_values("horizon_minutes").reset_index(drop=True)
    idxs = ordered.index[ordered["joint_interpretation"].isin(labels)].to_list()
    if not idxs:
        return ""

    ranges = []
    start = prev = idxs[0]
    for idx in idxs[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append((start, prev))
        start = prev = idx
    ranges.append((start, prev))

    labels_out = []
    for start, end in ranges:
        left = ordered.loc[start, "horizon_label"]
        right = ordered.loc[end, "horizon_label"]
        labels_out.append(left if start == end else f"{left} to {right}")
    return "; ".join(labels_out)


def interpretation_rows(ticker: str, combined_df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "ticker": ticker,
            "signal_type": "mean_reversion",
            "time_scale": compact_ranges(
                combined_df, {"mean-reverting", "weak mean-reverting", "push-response only: mean-reverting"}
            ),
        },
        {
            "ticker": ticker,
            "signal_type": "trend_following",
            "time_scale": compact_ranges(
                combined_df, {"trend-following", "weak trend-following", "push-response only: trend-following"}
            ),
        },
        {
            "ticker": ticker,
            "signal_type": "no_clear_inefficiency",
            "time_scale": compact_ranges(combined_df, {"no clear inefficiency"}),
        },
    ]
    return pd.DataFrame(rows)


def statistical_testing_table(combined_df: pd.DataFrame) -> pd.DataFrame:
    ordered = combined_df.sort_values("horizon_minutes")
    return pd.DataFrame(
        {
            "Horizon": ordered["horizon_label"],
            "VR": ordered["variance_ratio"].map(lambda x: f"{x:.4f}" if np.isfinite(x) else "n/a"),
            "VR robust p": ordered["vr_p_heteroskedastic"].map(p_label),
            "VR signal": ordered["vr_interpretation"],
            "PR beta": ordered["beta"].map(lambda x: f"{x:.4f}" if np.isfinite(x) else "n/a"),
            "PR beta p": ordered["beta_p_value"].map(p_label),
            "PR signal": ordered["pr_interpretation"],
            "Joint reading": ordered["joint_interpretation"],
        }
    )


def plot_variance_ratio(ticker: str, vr_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(vr_df["horizon_minutes"], vr_df["variance_ratio"], marker="o")
    significant = vr_df["vr_p_heteroskedastic"] < SIGNIFICANCE_LEVEL
    ax.scatter(
        vr_df.loc[significant, "horizon_minutes"],
        vr_df.loc[significant, "variance_ratio"],
        s=80,
        facecolors="none",
        edgecolors="black",
    )
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"{ticker} Variance Ratio")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Variance Ratio")
    ax.set_xscale("log")
    ax.set_xticks(vr_df["horizon_minutes"])
    ax.set_xticklabels(vr_df["horizon_label"], rotation=35, ha="right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "variance_ratio.png", dpi=160)
    plt.close(fig)


def plot_push_response(ticker: str, pr_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pr_df["horizon_minutes"], pr_df["beta"], marker="o")
    significant = pr_df["beta_p_value"] < SIGNIFICANCE_LEVEL
    ax.scatter(
        pr_df.loc[significant, "horizon_minutes"],
        pr_df.loc[significant, "beta"],
        s=80,
        facecolors="none",
        edgecolors="black",
    )
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"{ticker} Push-Response Beta")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Beta")
    ax.set_xscale("log")
    ax.set_xticks(pr_df["horizon_minutes"])
    ax.set_xticklabels(pr_df["horizon_label"], rotation=35, ha="right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "push_response_beta.png", dpi=160)
    plt.close(fig)


def plot_inefficiency(ticker: str, combined_df: pd.DataFrame, output_dir: Path) -> None:
    scores = combined_df["joint_interpretation"].map(interpretation_score)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.scatter(combined_df["horizon_minutes"], scores, s=90)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"{ticker} Inefficiency by Time Scale")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Signal")
    ax.set_xscale("log")
    ax.set_xticks(combined_df["horizon_minutes"])
    ax.set_xticklabels(combined_df["horizon_label"], rotation=35, ha="right")
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_yticklabels(
        ["mean-reverting", "weak mean-reverting", "no clear", "weak trend-following", "trend-following"]
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "inefficiency_timescale.png", dpi=160)
    plt.close(fig)


def save_market_outputs(
    market: MarketConfig,
    df: pd.DataFrame,
    vr_df: pd.DataFrame,
    pr_df: pd.DataFrame,
    combined_df: pd.DataFrame,
) -> pd.DataFrame:
    output_dir = OUTPUT_DIR / market.ticker
    output_dir.mkdir(parents=True, exist_ok=True)

    market_info = pd.DataFrame(
        [
            {
                "ticker": market.ticker,
                "name": market.name,
                "exchange": market.exchange,
                "currency": market.currency,
                "point_value": market.point_value,
                "tick_size": market.tick_size,
                "tick_value": market.tick_value,
                "slippage": market.slippage,
                "bars": len(df),
                "sample_start": df["DateTime"].iloc[0],
                "sample_end": df["DateTime"].iloc[-1],
            }
        ]
    )
    market_info.to_csv(output_dir / "market_info.csv", index=False)
    vr_df.to_csv(output_dir / "variance_ratio.csv", index=False)
    pr_df.to_csv(output_dir / "push_response.csv", index=False)
    combined_df.to_csv(output_dir / "random_walk_tests_combined.csv", index=False)
    statistical_testing_table(combined_df).to_csv(output_dir / "statistical_testing_table.csv", index=False)
    interp_df = interpretation_rows(market.ticker, combined_df)
    interp_df.to_csv(output_dir / "inefficiency_interpretation.csv", index=False)

    plot_variance_ratio(market.ticker, vr_df, output_dir)
    plot_push_response(market.ticker, pr_df, output_dir)
    plot_inefficiency(market.ticker, combined_df, output_dir)
    return interp_df


def run_market(ticker: str) -> pd.DataFrame:
    market = load_market_config(ticker)
    df = load_price_data(data_file(ticker))
    vr_df = run_variance_ratio(df)
    pr_df = run_push_response(df)
    combined_df = combine_tests(vr_df, pr_df)
    interp_df = save_market_outputs(market, df, vr_df, pr_df, combined_df)

    print(f"{market.ticker}: {len(df):,} bars, {df['DateTime'].iloc[0]} to {df['DateTime'].iloc[-1]}")
    print(statistical_testing_table(combined_df).to_string(index=False))
    print()
    return interp_df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    comparison = pd.concat([run_market(ticker) for ticker in MARKETS], ignore_index=True)
    comparison.to_csv(OUTPUT_DIR / "co_btc_inefficiency_comparison.csv", index=False)


if __name__ == "__main__":
    main()
