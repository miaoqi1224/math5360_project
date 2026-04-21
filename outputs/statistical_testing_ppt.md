# PPT Outline: Statistical Testing

## Slide 1. Statistical Testing Setup
- Market: CO (Brent Crude)
- Data frequency: 5-minute bars
- Sample: 2003-08-01 14:05:00 to 2026-04-10 19:30:00
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
- Consistent mean-reversion at: [60, 120, 480] minutes
- Mixed evidence at: [5, 15, 30, 240] minutes
- Very short horizons are closer to Random Walk

## Slide 5. Final Conclusion
- Brent crude shows horizon-dependent predictability
- The clearest inefficiency is mean-reversion
- Most reliable horizons: 60, 120, and 480 minutes
- This is the main takeaway for the statistical testing section
