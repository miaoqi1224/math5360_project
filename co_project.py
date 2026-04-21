from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import xlrd
except Exception:  # pragma: no cover
    xlrd = None


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

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
        vr_type = row["vr_interpretation"]
        pr_type = row["pr_interpretation"]
        if vr_type == pr_type and vr_type in {"trend-following", "mean-reverting"}:
            return vr_type
        if vr_type == "close to random walk" and pr_type == "close to random walk":
            return "close to random walk"
        return "mixed evidence"

    merged["joint_interpretation"] = merged.apply(classify, axis=1)
    return merged


def write_report_and_ppt(market: MarketConfig, df: pd.DataFrame, combined_df: pd.DataFrame) -> None:
    mean_reverting_horizons = combined_df.loc[
        combined_df["joint_interpretation"] == "mean-reverting", "horizon_minutes"
    ].tolist()
    mixed_horizons = combined_df.loc[
        combined_df["joint_interpretation"] == "mixed evidence", "horizon_minutes"
    ].tolist()

    report = f"""# Statistical Testing Report

## Data

We study the CO (Brent Crude) futures market using 5-minute data from {df["DateTime"].iloc[0]} to {df["DateTime"].iloc[-1]}. According to TF Data, the market trades on {market.exchange} in {market.currency}, with point value {market.point_value}, tick size {market.tick_size}, tick value {market.tick_value}, and suggested round-turn slippage {market.slippage}.

## Variance Ratio Test

The Variance Ratio (VR) test is used to compare the behavior of the CO time series against the Random Walk benchmark. If the VR value is close to 1, the series is approximately consistent with Random Walk. If VR is above 1, it suggests trend-following behavior. If VR is below 1, it suggests mean-reversion.

Our results show that the 5-minute, 15-minute, and 30-minute horizons are close to Random Walk. Starting from 60 minutes, the VR values are below 1 and remain below 1 at 120, 240, and 480 minutes. This indicates stronger mean-reverting behavior at medium and longer intraday horizons.

## Push-Response Test

The Push-Response (PR) test measures whether a future price change tends to continue or reverse a previous price move. A positive response beta suggests trend-following, while a negative response beta suggests mean-reversion.

The Push-Response results are mixed across time scales. At 5, 60, 120, and 480 minutes, the test indicates mean-reversion. At 15, 30, and 240 minutes, it indicates trend-following. Therefore, the PR test suggests that predictability in Brent crude depends on the time scale rather than following a single pattern across all horizons.

## Joint Inefficiency Interpretation

We combine the two tests by checking whether both point in the same direction. The strongest agreement appears at these horizons: {mean_reverting_horizons} minutes, where both tests support mean-reversion. The horizons with mixed evidence are: {mixed_horizons} minutes.

Overall, the current evidence suggests that Brent crude is close to Random Walk at very short horizons, but displays clearer mean-reverting inefficiency at medium horizons, especially around 60 to 120 minutes, and again at 480 minutes.

## Final Conclusion

The statistical testing section does not support a uniform trend-following interpretation across all horizons. Instead, the market shows horizon-dependent predictability, with the clearest and most consistent inefficiency being mean-reversion at medium time scales.
"""

    ppt = f"""# PPT Outline: Statistical Testing

## Slide 1. Statistical Testing Setup
- Market: CO (Brent Crude)
- Data frequency: 5-minute bars
- Sample: {df["DateTime"].iloc[0]} to {df["DateTime"].iloc[-1]}
- Tests used: Variance Ratio and Push-Response
- Goal: identify inefficiency type and its time-scale location

## Slide 2. Variance Ratio Results
- 5, 15, 30 minutes: close to Random Walk
- 60, 120, 240, 480 minutes: VR < 1
- Interpretation: stronger mean-reversion from 60 minutes onward
- Figure: outputs/variance_ratio.png

## Slide 3. Push-Response Results
- 5, 60, 120, 480 minutes: mean-reverting
- 15, 30, 240 minutes: trend-following
- Interpretation: predictability depends on horizon
- Figure: outputs/push_response_beta.png

## Slide 4. Joint Time-Scale Interpretation
- Consistent mean-reversion at: {mean_reverting_horizons} minutes
- Mixed evidence at: {mixed_horizons} minutes
- Very short horizons are closer to Random Walk

## Slide 5. Final Conclusion
- Brent crude shows horizon-dependent predictability
- The clearest inefficiency is mean-reversion
- Most reliable horizons: 60, 120, and 480 minutes
- This is the main takeaway for the statistical testing section
"""

    (OUTPUT_DIR / "statistical_testing_report.md").write_text(report, encoding="utf-8")
    (OUTPUT_DIR / "statistical_testing_ppt.md").write_text(ppt, encoding="utf-8")


def save_outputs(market: MarketConfig, df: pd.DataFrame, vr_df: pd.DataFrame, pr_df: pd.DataFrame, combined_df: pd.DataFrame) -> None:
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

    summary = f"""# Statistical Testing Summary

- Market: {market.ticker} ({market.name})
- Exchange: {market.exchange}
- Currency: {market.currency}
- Point value: {market.point_value}
- Tick size: {market.tick_size}
- Tick value: {market.tick_value}
- Slippage: {market.slippage}
- Sample bars: {len(df)}
- Sample start: {df["DateTime"].iloc[0]}
- Sample end: {df["DateTime"].iloc[-1]}

## Joint Results

```text
{combined_df.to_string(index=False, float_format=lambda x: f"{x:.6f}")}
```
"""
    (OUTPUT_DIR / "summary.md").write_text(summary, encoding="utf-8")
    write_report_and_ppt(market, df, combined_df)


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
    save_outputs(market, df, vr_df, pr_df, combined_df)
    print_console_summary(market, df, combined_df)


if __name__ == "__main__":
    main()
