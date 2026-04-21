# Statistical Testing Report

## Data

We study the CO (Brent Crude) futures market using 5-minute data from 2003-08-01 14:05:00 to 2026-04-10 19:30:00. According to TF Data, the market trades on ICE in USD, with point value 1000.0, tick size 0.01, tick value 10.0, and suggested round-turn slippage 48.0.

## Variance Ratio Test

The Variance Ratio (VR) test is used to compare the behavior of the CO time series against the Random Walk benchmark. If the VR value is close to 1, the series is approximately consistent with Random Walk. If VR is above 1, it suggests trend-following behavior. If VR is below 1, it suggests mean-reversion.

Our results show that the 5-minute, 15-minute, and 30-minute horizons are close to Random Walk. Starting from 60 minutes, the VR values are below 1 and remain below 1 at 120, 240, and 480 minutes. This indicates stronger mean-reverting behavior at medium and longer intraday horizons.

## Push-Response Test

The Push-Response (PR) test measures whether a future price change tends to continue or reverse a previous price move. A positive response beta suggests trend-following, while a negative response beta suggests mean-reversion.

The Push-Response results are mixed across time scales. At 5, 60, 120, and 480 minutes, the test indicates mean-reversion. At 15, 30, and 240 minutes, it indicates trend-following. Therefore, the PR test suggests that predictability in Brent crude depends on the time scale rather than following a single pattern across all horizons.

## Joint Inefficiency Interpretation

We combine the two tests by checking whether both point in the same direction. The strongest agreement appears at these horizons: [60, 120, 480] minutes, where both tests support mean-reversion. The horizons with mixed evidence are: [5, 15, 30, 240] minutes.

Overall, the current evidence suggests that Brent crude is close to Random Walk at very short horizons, but displays clearer mean-reverting inefficiency at medium horizons, especially around 60 to 120 minutes, and again at 480 minutes.

## Final Conclusion

The statistical testing section does not support a uniform trend-following interpretation across all horizons. Instead, the market shows horizon-dependent predictability, with the clearest and most consistent inefficiency being mean-reversion at medium time scales.
