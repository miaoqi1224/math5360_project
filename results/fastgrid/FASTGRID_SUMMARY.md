# Fast-Grid Pilot Summary (CO + BTC)

This folder stores a quick, reproducible pilot run used to shortlist promising settings before expensive full-grid jobs.

## CO (Primary Market)
### Week2 Fast-Grid Optimization
- Best point: `L=500`, `S=0.005`
- Best `return_to_dd`: `85.9793`; in-sample Sharpe at best point: `2.6198`
- Artifacts: `co/week2/optimization_results.csv`, `co/week2/optimization_results_full.csv`, heatmaps PNGs

### Week3 Fast-Grid Rolling OOS
- Rolling windows: `76`
- Parameter variation: `best_L` unique=5, `best_S` unique=3
- First-window best pair: `L=500`, `S=0.020`
- Artifacts: `co/week3/oos_equity.csv`, `co/week3/rolling_parameters.csv`, `co/week3/oos_trades.csv`, `co/week3/rolling_oos_equity_dd.png`

### Week4 Performance Snapshot (from fast-grid Week3 outputs)
- Total return: `361770.90`
- Max drawdown: `-15571.90`
- Sharpe ratio: `1.7303`
- Profit factor: `1.2302`
- Total trades: `1416`
- Artifacts: all files under `co/week4/`

## BTC (Secondary Market)
### Week2 Fast-Grid Optimization
- Best point: `L=500`, `S=0.005`
- Best `return_to_dd`: `23.6157`; in-sample Sharpe at best point: `2.4171`
- Artifacts: `btc/week2/optimization_results.csv`, `btc/week2/optimization_results_full.csv`, heatmaps PNGs

### Week3 Fast-Grid Rolling OOS
- Rolling windows: `17`
- Parameter variation: `best_L` unique=2, `best_S` unique=1
- First-window best pair: `L=500`, `S=0.010`
- Artifacts: `btc/week3/oos_equity.csv`, `btc/week3/rolling_parameters.csv`, `btc/week3/oos_trades.csv`, `btc/week3/rolling_oos_equity_dd.png`

### Week4 Performance Snapshot (from fast-grid Week3 outputs)
- Total return: `3310417.50`
- Max drawdown: `-465806.25`
- Sharpe ratio: `2.1721`
- Profit factor: `1.1527`
- Total trades: `1632`
- Artifacts: all files under `btc/week4/`

## Conclusions (Pilot Only)
- Fast-grid quickly identifies promising regions; both markets favored short channel lengths in this coarse search.
- CO still shows stronger `return_to_dd` in this pilot; BTC shows much larger absolute return with materially larger drawdown.
- These results are exploratory and should not be used as final submission metrics; rerun full-grid for final report tables.
