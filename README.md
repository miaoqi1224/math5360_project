# MATH 5360 / quant project - what I actually built (week 1 + week 2)

Hi, this repo is basically two chunks of work on the same HO futures data (5 min bars). Week 1 is stats only. Week 2 bolts onto my python backtest from class and does a parameter sweep. Nothing here is meant to be trading advice.

---

## What's in the folder (the stuff I touch a lot)

- **main.py** - python version of the channel breakout + trailing stop thing from matlab. Has `rolling_hh_ll` and `run_backtest` unchanged, then I added the week 2 helper functions at the bottom (`run_strategy`, `compute_metrics`, etc.).
- **main.m** - original matlab if you still have a copy (not always in the zip depending what got uploaded).
- **week1_timeseries_analysis.py** - builds datetime index, log returns, variance ratio test, push-response thing, plots.
- **week2_optimization.py** - just calls into main.py and runs the grid, saves csv + heatmaps.
- **HO-5minHLV.csv** - the data file (big). You need it in the same folder or change the path in the scripts.
- **requirements.txt** - pip stuff. Week 1 needs scipy too for skew/kurtosis and the VR p-values.

Other files like `optimization_results*.csv` and the png heatmaps / week1 figs show up after you run things. There's also team stuff from the shared branch (`co_project.py`, `outputs/`, etc.) - I left those alone.

---

## Week 1 - what it does

I take Date + Time, merge into one datetime index, sort, fill missing OHLC forward, and drop rows where close is 0 or negative (otherwise log returns blow up).

Then: log returns, mean/std/skew/excess kurtosis, variance ratio for lags 2,4,8,16,32 (overlapping version, same idea as Lo-MacKinlay notes from class). Push-response: I flag the top 5% of |r| as "big moves", split into up vs down, and average the cumulative log return over the next 1,5,10,20 bars.

Run it:

```bash
pip install -r requirements.txt
python week1_timeseries_analysis.py
```

If you're on a server without a display, set `MPLBACKEND=Agg` and it saves pngs next to the script instead of popping windows.

Notebook version is basically:

```python
from week1_timeseries_analysis import run_week1_pipeline
out = run_week1_pipeline(df, show_plots=True)
print(out["summary_text"])
```

Important: week 1 never imports the trading rules from main.py. It's only looking at returns.

---

## Week 2 - what it does

I didn't want to rewrite the inner loop of the strategy, so `run_strategy` is literally `rolling_hh_ll` then `run_backtest` with the same bars_back / slpg / pv / e0 defaults as before. `compute_metrics` spits out return, max drawdown (min of DD series), return divided by abs(max DD) as my main score, a rough sharpe on bar pnl after the warm-up period, trade count, etc.

Grid I used: L from 500 to 10000 step 500, S from 0.005 to 0.05 step 0.005. That's 200 backtests - on my laptop it was like a bit over a minute, yours will vary.

```bash
cd path/to/this/folder
python week2_optimization.py
```

Writes `optimization_results.csv` (sorted by return/dd) and a fuller csv, plus two matplotlib heatmaps (I didn't use seaborn bc the assignment said not to).

Honest caveat: this is all in-sample on one long series. If I were writing the paper I'd spend more time on why that might overfit and what I'd do for a real holdout.

---

## Running the original matlab-style demo

```bash
python main.py
```

That's the in-sample / out-of-sample printout + plots from the port of main.m. Separate from week1 and week2 entry points.

---

## If something breaks

- "No module named scipy" -> you forgot scipy for week 1.
- csv not found -> check you're in the right directory or the filename matches.
- heatmaps look weird with all nan -> usually means something went wrong in metrics (e.g. no drawdown) for that cell; check the printed table.

---

Simon

*(If I change the grid later I'll try to remember to edit this readme so it doesn't lie about the defaults.)*
