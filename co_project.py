from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import xlrd
except Exception:
    xlrd = None


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

# Prefer files in the repo itself, then fall back to the earlier working folder.
DATA_CANDIDATES = [
    BASE_DIR / "CO-5minHLV.csv",
    Path("/Users/regina/Desktop/5360 project/CO-5minHLV.csv"),
]

TF_DATA_CANDIDATES = [
    BASE_DIR / "TF Data.xls",
    Path("/Users/regina/Desktop/5360 project/TF Data.xls"),
]

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


def resolve_existing_path(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find any of these paths: {candidates}")


def load_market_config() -> MarketConfig:
    # Use hard-coded CO specs if TF Data.xls is unavailable locally.
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

    if xlrd is None:
        return fallback

    tf_path = resolve_existing_path(TF_DATA_CANDIDATES)
    book = xlrd.open_workbook(str(tf_path))
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


def load_data() -> pd.DataFrame:
    data_path = resolve_existing_path(DATA_CANDIDATES)
    df = pd.read_csv(data_path)
    df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%m/%d/%Y %H:%M")
    df = df.sort_values("DateTime").reset_index(drop=True)
    return df


def compute_log_returns(close: np.ndarray) -> np.ndarray:
    return np.diff(np.log(close))


def variance_ratio(one_bar_returns: np.ndarray, q: int) -> float:
    # Compare q-bar return variance to q times the 1-bar variance.
    if q <= 0 or len(one_bar_returns) <= q:
        return float("nan")

    var_1 = np.var(one_bar_returns, ddof=1)
    if var_1 == 0:
        return float("nan")

    aggregated = np.convolve(one_bar_returns, np.ones(q), mode="valid")
    var_q = np.var(aggregated, ddof=1)
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
    returns = compute_log_returns(close)
    rows = []
    for horizon in VR_HORIZONS:
        vr = variance_ratio(returns, horizon)
        rows.append(
            {
                "horizon_bars": horizon,
                "horizon_minutes": horizon * 5,
                "variance_ratio": vr,
                "vr_interpretation": interpret_vr(vr),
            }
        )
    return pd.DataFrame(rows)


def push_response_for_horizon(close: np.ndarray, horizon: int) -> tuple[float, float]:
    # Build adjacent non-overlapping push and response windows of equal length.
    pushes = close[horizon::horizon] - close[:-horizon:horizon]
    responses = close[2 * horizon :: horizon] - close[horizon:-horizon:horizon]

    n = min(len(pushes), len(responses))
    pushes = pushes[:n]
    responses = responses[:n]
    if n < 2:
        return float("nan"), float("nan")

    push_var = np.var(pushes, ddof=1)
    beta = float(np.cov(pushes, responses, ddof=1)[0, 1] / push_var) if push_var > 0 else float("nan")
    signed_response = float(np.mean(np.sign(pushes) * responses))
    return beta, signed_response


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
    for horizon in PR_HORIZONS:
        beta, signed_response = push_response_for_horizon(close, horizon)
        rows.append(
            {
                "horizon_bars": horizon,
                "horizon_minutes": horizon * 5,
                "beta": beta,
                "signed_response": signed_response,
                "pr_interpretation": interpret_push_response(beta),
            }
        )
    return pd.DataFrame(rows)


def combine_rw_tests(vr_df: pd.DataFrame, pr_df: pd.DataFrame) -> pd.DataFrame:
    merged = vr_df.merge(pr_df, on=["horizon_bars", "horizon_minutes"], how="inner")

    def classify(row: pd.Series) -> str:
        # Treat agreement between both tests as the cleanest interpretation.
        vr_type = row["vr_interpretation"]
        pr_type = row["pr_interpretation"]
        if vr_type == pr_type and vr_type in {"trend-following", "mean-reverting"}:
            return vr_type
        if vr_type == "close to random walk" and pr_type == "close to random walk":
            return "close to random walk"
        return "mixed evidence"

    merged["joint_interpretation"] = merged.apply(classify, axis=1)
    return merged


def save_outputs(vr_df: pd.DataFrame, pr_df: pd.DataFrame, combined_df: pd.DataFrame) -> None:
    # Save tables first, then export the two figures used in the PPT section.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    vr_df.to_csv(OUTPUT_DIR / "variance_ratio.csv", index=False)
    pr_df.to_csv(OUTPUT_DIR / "push_response.csv", index=False)
    combined_df.to_csv(OUTPUT_DIR / "random_walk_tests_combined.csv", index=False)

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
    ax.plot(pr_df["horizon_minutes"], pr_df["beta"], marker="o")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("CO Push-Response Beta by Horizon")
    ax.set_xlabel("Horizon (minutes)")
    ax.set_ylabel("Push-Response Beta")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "push_response_beta.png", dpi=160)
    plt.close(fig)


def print_console_summary(market: MarketConfig, df: pd.DataFrame, combined_df: pd.DataFrame) -> None:
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
    print(f"Outputs written to: {OUTPUT_DIR}")


def main() -> None:
    market = load_market_config()
    df = load_data()
    vr_df = run_variance_ratio_scan(df)
    pr_df = run_push_response_scan(df)
    combined_df = combine_rw_tests(vr_df, pr_df)
    save_outputs(vr_df, pr_df, combined_df)
    print_console_summary(market, df, combined_df)


if __name__ == "__main__":
    main()
