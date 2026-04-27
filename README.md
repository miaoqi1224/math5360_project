# MATH GR5360 Final Project

Group #5 primary market:

- `CO`: Brent Crude Oil Futures

Secondary market:

- `BTC`: CME Bitcoin Futures

This repository runs the same analysis pipeline for both markets.

## Main Script

- `co_project.py`

The script loads 5-minute OHLC data, reads market contract parameters from `TF Data.xls`, runs statistical random-walk tests, and runs the channel walk-forward backtest.

## Data Files

- `CO-5minHLV.csv`
- `BTC-5minHLV.csv`
- `TF Data.xls`

## Run

Run both markets:

```bash
python3 co_project.py
```

Run one market:

```bash
python3 co_project.py CO
python3 co_project.py BTC
```

## Outputs

CO outputs are saved under:

- `outputs/CO/`

BTC outputs are saved under:

- `outputs/BTC/`

Each market output folder contains:

- `variance_ratio.csv`
- `push_response.csv`
- `random_walk_tests_combined.csv`
- `statistical_testing_table.csv`
- `variance_ratio.png`
- `push_response_beta.png`
- `inefficiency_timescale.png`
- `walk_forward_oos_equity.csv`
- `walk_forward_oos_equity.png`
- `walk_forward_quarterly_parameters.csv`

## Notes

- `CO` is the assigned primary market.
- `BTC` is used as the secondary market.
- BTC has a shorter sample because CME Bitcoin Futures start later than Brent Crude Oil Futures.
- `QUICK_MODE = True` uses a smaller parameter grid for faster runs.
- Set `QUICK_MODE = False` in `co_project.py` to use the full project grid.
