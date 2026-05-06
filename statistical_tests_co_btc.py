from __future__ import annotations

# Subgroup 3 main file: CO and BTC statistical tests.

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import xlrd
except Exception:  # pragma: no cover
    xlrd = None


PROJECT_DIR = Path(__file__).resolve().parent
TF_DATA_FILE = PROJECT_DIR / "TF Data.xls"
OUTPUT_DIR = PROJECT_DIR / "outputs"

MARKETS = ("CO", "BTC")
BASE_BAR_MINUTES = 5

VR_BASE_HORIZONS = (1, 6, 12)  # 5 min, 30 min, 1 hr base shifts
VR_Q_MULTIPLES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 192)

PR_HORIZONS = (1, 2, 3, 6, 12, 24, 48, 96, 192)
PR_PLOT_HORIZONS = (1, 3, 12)
PR_BINS = 11
PR_TRIM = 0.005
MIN_BIN_COUNT = 50
SIGNAL_TOLERANCE = 0.01


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


def horizon_label(bars: int) -> str:
    minutes = bars * BASE_BAR_MINUTES
    if minutes < 60:
        return f"{minutes:.0f} min"
    hours = minutes / 60.0
    if hours < 24:
        return f"{hours:.1f} hr"
    return f"{hours:.1f} hr (~{hours / 24.0:.1f} days)"


def price_change(close: np.ndarray, bars: int) -> np.ndarray:
    return close[bars:] - close[:-bars]


def direction_label(value: float, tolerance: float = SIGNAL_TOLERANCE) -> str:
    if not np.isfinite(value):
        return "insufficient data"
    if abs(value) <= tolerance:
        return "no clear"
    if value < 0:
        return "mean-reverting"
    return "trend-following"


def joint_label(vr_signal: str, pr_signal: str) -> str:
    directional = {"mean-reverting", "trend-following"}
    if vr_signal == pr_signal and vr_signal in directional:
        return vr_signal
    if vr_signal in directional and pr_signal == "no clear":
        return f"weak {vr_signal}"
    if pr_signal in directional and vr_signal == "no clear":
        return f"weak {pr_signal}"
    if vr_signal == "baseline" and pr_signal in directional:
        return f"push-response only: {pr_signal}"
    if vr_signal in directional and pr_signal == "insufficient data":
        return f"weak {vr_signal}"
    if pr_signal in directional and vr_signal == "insufficient data":
        return f"weak {pr_signal}"
    if vr_signal in directional and pr_signal in directional:
        return "mixed evidence"
    return "no clear"


def variance_ratio_curve(close: np.ndarray, base_bars: int) -> pd.DataFrame:
    base_changes = price_change(close, base_bars)
    base_var = float(np.var(base_changes, ddof=1))
    rows = []

    for q_multiple in VR_Q_MULTIPLES:
        horizon_bars = base_bars * q_multiple
        if len(close) <= horizon_bars or base_var <= 0:
            continue
        horizon_changes = price_change(close, horizon_bars)
        vr = float(np.var(horizon_changes, ddof=1) / (q_multiple * base_var))
        rows.append(
            {
                "base_bars": base_bars,
                "base_label": horizon_label(base_bars),
                "q_multiple": q_multiple,
                "horizon_bars": horizon_bars,
                "horizon_label": horizon_label(horizon_bars),
                "horizon_minutes": horizon_bars * BASE_BAR_MINUTES,
                "variance_ratio": vr,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["vr_curve_change"] = out["variance_ratio"].diff()
    out.loc[out["q_multiple"] == 1, "vr_curve_change"] = np.nan
    out["vr_signal"] = out["vr_curve_change"].map(direction_label)
    out.loc[out["q_multiple"] == 1, "vr_signal"] = "baseline"
    return out


def run_variance_ratio(close: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves = pd.concat([variance_ratio_curve(close, q) for q in VR_BASE_HORIZONS], ignore_index=True)
    main = curves.loc[curves["base_bars"] == 1].copy()
    return main, curves


def push_response_bins(close: np.ndarray, bars: int) -> pd.DataFrame:
    push = close[bars::bars] - close[:-bars:bars]
    response = close[2 * bars :: bars] - close[bars:-bars:bars]
    n = min(len(push), len(response))
    push = push[:n].astype(float)
    response = response[:n].astype(float)

    scale = float(np.std(push, ddof=1)) if n > 1 else float("nan")
    if n < MIN_BIN_COUNT * 3 or not np.isfinite(scale) or scale <= 0:
        return pd.DataFrame()

    x = push / scale
    y = response / scale
    lo, hi = np.quantile(x, [PR_TRIM, 1.0 - PR_TRIM])
    keep = (x >= lo) & (x <= hi)
    x = x[keep]
    y = y[keep]
    if len(x) < MIN_BIN_COUNT * 3:
        return pd.DataFrame()

    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, PR_BINS + 1)))
    if len(edges) < 4:
        return pd.DataFrame()

    bin_id = np.digitize(x, edges[1:-1], right=True)
    rows = []
    for i in range(len(edges) - 1):
        mask = bin_id == i
        count = int(mask.sum())
        if count < MIN_BIN_COUNT:
            continue
        rows.append(
            {
                "horizon_bars": bars,
                "horizon_label": horizon_label(bars),
                "horizon_minutes": bars * BASE_BAR_MINUTES,
                "bin": i + 1,
                "n": count,
                "push_mean": float(np.mean(x[mask])),
                "response_mean": float(np.mean(y[mask])),
                "response_se": float(np.std(y[mask], ddof=1) / np.sqrt(count)),
            }
        )
    return pd.DataFrame(rows)


def push_response_summary(curve: pd.DataFrame, bars: int) -> dict[str, float | int | str]:
    base = {
        "horizon_bars": bars,
        "horizon_label": horizon_label(bars),
        "horizon_minutes": bars * BASE_BAR_MINUTES,
    }
    if curve.empty or len(curve) < 3:
        return {
            **base,
            "n_push_response_pairs": 0,
            "response_curve_slope": float("nan"),
            "signed_conditional_response": float("nan"),
            "pr_signal": "insufficient data",
        }

    x = curve["push_mean"].to_numpy(dtype=float)
    y = curve["response_mean"].to_numpy(dtype=float)
    w = curve["n"].to_numpy(dtype=float)
    x_bar = float(np.average(x, weights=w))
    y_bar = float(np.average(y, weights=w))
    centered_x = x - x_bar
    denom = float(np.sum(w * centered_x**2))
    slope = float(np.sum(w * centered_x * (y - y_bar)) / denom) if denom > 0 else float("nan")
    signed_response = float(np.average(np.sign(x) * y, weights=w))

    return {
        **base,
        "n_push_response_pairs": int(curve["n"].sum()),
        "response_curve_slope": slope,
        "signed_conditional_response": signed_response,
        "pr_signal": direction_label(slope),
    }


def run_push_response(close: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves = []
    summaries = []
    for bars in PR_HORIZONS:
        curve = push_response_bins(close, bars)
        if not curve.empty:
            curves.append(curve)
        summaries.append(push_response_summary(curve, bars))
    curve_df = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    summary_df = pd.DataFrame(summaries)
    return curve_df, summary_df


def combined_table(vr_main: pd.DataFrame, pr_summary: pd.DataFrame) -> pd.DataFrame:
    merged = vr_main.merge(
        pr_summary,
        on=("horizon_bars", "horizon_label", "horizon_minutes"),
        how="inner",
    )
    merged["joint_reading"] = [joint_label(vr, pr) for vr, pr in zip(merged["vr_signal"], merged["pr_signal"])]
    return merged


def compact_ranges(df: pd.DataFrame, labels: set[str]) -> str:
    selected = df.loc[df["joint_reading"].isin(labels)].sort_values("horizon_minutes").reset_index(drop=True)
    if selected.empty:
        return ""

    ranges = []
    start = prev = 0
    for i in range(1, len(selected)):
        current_minutes = selected.loc[i, "horizon_minutes"]
        previous_minutes = selected.loc[i - 1, "horizon_minutes"]
        if current_minutes <= previous_minutes * 2.1:
            prev = i
            continue
        ranges.append((start, prev))
        start = prev = i
    ranges.append((start, prev))

    output = []
    for start, end in ranges:
        left = selected.loc[start, "horizon_label"]
        right = selected.loc[end, "horizon_label"]
        output.append(left if start == end else f"{left} to {right}")
    return "; ".join(output)


def inefficiency_summary(ticker: str, combined: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": ticker,
                "signal_type": "mean_reversion",
                "time_scale": compact_ranges(
                    combined,
                    {"mean-reverting", "weak mean-reverting", "push-response only: mean-reverting"},
                ),
            },
            {
                "ticker": ticker,
                "signal_type": "trend_following",
                "time_scale": compact_ranges(
                    combined,
                    {"trend-following", "weak trend-following", "push-response only: trend-following"},
                ),
            },
            {
                "ticker": ticker,
                "signal_type": "no_clear",
                "time_scale": compact_ranges(combined, {"no clear"}),
            },
        ]
    )


def presentation_table(combined: pd.DataFrame) -> pd.DataFrame:
    ordered = combined.sort_values("horizon_minutes")
    return pd.DataFrame(
        {
            "Horizon": ordered["horizon_label"],
            "VR": ordered["variance_ratio"].map(lambda x: f"{x:.4f}"),
            "VR curve change": ordered["vr_curve_change"].map(lambda x: f"{x:.4f}" if np.isfinite(x) else "n/a"),
            "VR signal": ordered["vr_signal"],
            "Push-response slope": ordered["response_curve_slope"].map(
                lambda x: f"{x:.4f}" if np.isfinite(x) else "n/a"
            ),
            "Push-response signal": ordered["pr_signal"],
            "Joint reading": ordered["joint_reading"],
        }
    )


def plot_variance_ratio(ticker: str, curves: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for base_label, curve in curves.groupby("base_label", sort=False):
        curve = curve.sort_values("horizon_minutes")
        ax.plot(curve["horizon_minutes"], curve["variance_ratio"], marker="o", linewidth=1.7, label=base_label)

    ticks = sorted(curves["horizon_minutes"].unique())
    labels = curves.drop_duplicates("horizon_minutes").set_index("horizon_minutes")["horizon_label"].to_dict()
    ax.set_title(f"{ticker} Variance Ratio Curve")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("VR(q)")
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([labels[tick] for tick in ticks], rotation=35, ha="right")
    ax.legend(title="Base shift")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "variance_ratio.png", dpi=160)
    plt.close(fig)


def plot_push_response(ticker: str, curves: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    max_abs_push = 3.0
    for bars in PR_PLOT_HORIZONS:
        curve = curves.loc[curves["horizon_bars"] == bars].sort_values("push_mean")
        if curve.empty:
            continue
        ax.plot(curve["push_mean"], curve["response_mean"], marker="o", linewidth=1.7, label=horizon_label(bars))
        max_abs_push = max(max_abs_push, float(np.nanmax(np.abs(curve["push_mean"]))))

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)
    max_abs_push = min(max_abs_push * 1.05, 4.0)
    ax.set_xlim(-max_abs_push, max_abs_push)
    ax.set_title(f"{ticker} Push-Response Diagram")
    ax.set_xlabel("Push x: previous price change / sigma")
    ax.set_ylabel("Average response R(x): next price change / sigma")
    ax.legend(title="Horizon")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "push_response_diagram.png", dpi=160)
    plt.close(fig)


def plot_inefficiency(ticker: str, combined: pd.DataFrame, output_dir: Path) -> None:
    scores = {
        "mean-reverting": -2,
        "weak mean-reverting": -1,
        "push-response only: mean-reverting": -1,
        "mixed evidence": 0,
        "no clear": 0,
        "weak trend-following": 1,
        "push-response only: trend-following": 1,
        "trend-following": 2,
    }
    y = combined["joint_reading"].map(scores)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.scatter(combined["horizon_minutes"], y, s=90)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"{ticker} Inefficiency by Time Scale")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Signal")
    ax.set_xscale("log")
    ax.set_xticks(combined["horizon_minutes"])
    ax.set_xticklabels(combined["horizon_label"], rotation=35, ha="right")
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_yticklabels(["mean-reverting", "weak mean-reverting", "no clear", "weak trend-following", "trend-following"])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "inefficiency_timescale.png", dpi=160)
    plt.close(fig)


def save_market_outputs(market: MarketConfig, df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].to_numpy(dtype=float)
    vr_main, vr_curves = run_variance_ratio(close)
    pr_curves, pr_summary = run_push_response(close)
    combined = combined_table(vr_main, pr_summary)
    summary = inefficiency_summary(market.ticker, combined)

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
    vr_main.to_csv(output_dir / "variance_ratio.csv", index=False)
    vr_curves.to_csv(output_dir / "variance_ratio_curves.csv", index=False)
    pr_curves.to_csv(output_dir / "push_response.csv", index=False)
    pr_summary.to_csv(output_dir / "push_response_summary.csv", index=False)
    combined.to_csv(output_dir / "random_walk_tests_combined.csv", index=False)
    presentation_table(combined).to_csv(output_dir / "statistical_testing_table.csv", index=False)
    summary.to_csv(output_dir / "inefficiency_interpretation.csv", index=False)

    plot_variance_ratio(market.ticker, vr_curves, output_dir)
    plot_push_response(market.ticker, pr_curves, output_dir)
    plot_inefficiency(market.ticker, combined, output_dir)

    old_beta_plot = output_dir / "push_response_beta.png"
    if old_beta_plot.exists():
        old_beta_plot.unlink()

    print(f"{market.ticker}: {len(df):,} bars, {df['DateTime'].iloc[0]} to {df['DateTime'].iloc[-1]}")
    print(presentation_table(combined).to_string(index=False))
    print()
    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    for ticker in MARKETS:
        market = load_market_config(ticker)
        df = load_price_data(data_file(ticker))
        summaries.append(save_market_outputs(market, df))
    pd.concat(summaries, ignore_index=True).to_csv(OUTPUT_DIR / "co_btc_inefficiency_comparison.csv", index=False)


if __name__ == "__main__":
    main()
