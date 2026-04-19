# MATH GR5360 Final Project

Files in this folder:

- `co_project.py`: main project script
- `CO-5minHLV.csv`: Brent crude 5-minute data
- `TF Data.xls`: contract facts and slippage reference
- `CO DES.gif`, `CO CT.gif`, `CO GPO.gif`: Bloomberg reference screens
- `main.m`, `ezread.m`: professor-provided Matlab reference

## What the script does

`co_project.py` currently does four things:

1. Loads the Brent crude data and market facts.
2. Runs the two Random Walk tests used in class:
   Variance Ratio and Push-Response.
3. Runs a walk-forward optimization / out-of-sample backtest for the professor's
   basic trend-following channel system.
4. Saves CSV tables and PNG charts into `outputs/`.

## How to run

```bash
cd "/Users/regina/Desktop/5360 project"
python3 co_project.py
```

## GitHub sharing

If you want to publish this folder to GitHub after authenticating with GitHub CLI:

```bash
cd "/Users/regina/Desktop/5360 project"
./publish_github.sh
```

By default the script creates a private repository named `gr5360-co-project` under the
currently authenticated GitHub account and pushes the local `main` branch.

## Quick mode vs full mode

The professor's full optimization grid is large:

- `ChnLen = 500, 510, ..., 10000`
- `StpPct = 0.005, 0.006, ..., 0.100`

The script currently defaults to `QUICK_MODE = True` so it stays practical for daily work.
To run the professor-style full grid, open `co_project.py` and change:

```python
QUICK_MODE = False
```

## Outputs

The script writes:

- `outputs/variance_ratio.csv`
- `outputs/push_response.csv`
- `outputs/random_walk_tests_combined.csv`
- `outputs/walk_forward_quarterly_parameters.csv`
- `outputs/walk_forward_oos_equity.csv`
- `outputs/variance_ratio.png`
- `outputs/push_response_beta.png`
- `outputs/walk_forward_oos_equity.png`
- `outputs/summary.md`
