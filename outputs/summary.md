# Statistical Testing Summary

- Market: CO (Brent Crude)
- Exchange: ICE
- Currency: USD
- Point value: 1000.0
- Tick size: 0.01
- Tick value: 10.0
- Slippage: 48.0
- Sample bars: 384306
- Sample start: 2003-08-01 14:05:00
- Sample end: 2026-04-10 19:30:00

## Joint Results

```text
 horizon_bars  horizon_minutes  variance_ratio    vr_interpretation      beta  signed_response pr_interpretation joint_interpretation
            1                5        1.000000 close to random walk -0.004048        -0.001431    mean-reverting       mixed evidence
            3               15        0.985077 close to random walk  0.001472        -0.000354   trend-following       mixed evidence
            6               30        0.984978 close to random walk  0.001502         0.001241   trend-following       mixed evidence
           12               60        0.976802       mean-reverting -0.014484         0.002436    mean-reverting       mean-reverting
           24              120        0.963320       mean-reverting -0.014114        -0.002873    mean-reverting       mean-reverting
           48              240        0.960443       mean-reverting  0.007588         0.020177   trend-following       mixed evidence
           96              480        0.965787       mean-reverting -0.010255        -0.018473    mean-reverting       mean-reverting
```
