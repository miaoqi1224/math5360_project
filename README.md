# MATH GR5360 — Final Project (Group 5, CO / Brent crude)

This repository implements the **Final Project** spine for **Weeks 1–3** on the **primary market CO (Brent)**: 5-minute OHLC data, the course **Channel + trailing stop (WithDDControl)**-style system in Python, the two statistical tests, the **full PDF parameter grid** machinery, and **rolling walk-forward OOS**. A **secondary market** can be added later per the syllabus.

**Disclaimer:** The syllabus states you are not graded on how much money the strategy makes. All numbers here are for coursework and reporting only, not investment advice.

---

## 1. Environment

```bash
cd c:\Users\simon\OneDrive\desktop\5260\final
pip install -r requirements.txt
```

- **Week 1** needs **scipy** (variance-ratio p-values, etc.).
- Headless runs: set `MPLBACKEND=Agg` so figures save as PNG instead of opening GUI windows.

---

## 2. Repository layout

| Path | Role |
|------|------|
| `CO.csv` | Group 5 primary data: Brent (CO) 5-minute bars (`Date,Time,Open,High,Low,Close,Volume`). |
| `group5_config.py` | Defaults: `PRIMARY_DATA_FILE`, PDF **ChnLen / StpPct** grid builders, `DEFAULT_SLPG` / `DEFAULT_PV`, optional `group5_overrides.json`. |
| `group5_overrides.example.json` | Copy to `group5_overrides.json` and fill **Slpg** (column V) and **Point Value** (column H) for **CO** from **TF Data.xls**. |
| `main.py` | Data load, `rolling_hh_ll`, `run_backtest` (with position path), `run_strategy`, `optimize_parameters` (full grid when grids are `None`), `rolling_backtest`, plots, and `main()` demo. |
| `week1_timeseries_analysis.py` | Descriptive stats, **Variance Ratio**, **Push-Response**, figures + narrative text. |
| `week2_optimization.py` | **In-sample** grid over `(L,S)` on the full series; objective **return / \|max drawdown\|`**; CSVs + heatmaps. |
| `week3_rolling_oos.py` | **~4y IS / ~3m OOS** rolling walk-forward; per-window optimal `(L,S)`; **stitched OOS equity**, per-window metrics, **`oos_trades.csv`** (completed round-turns in OOS). |
| `week1_run.log`, `week2_run.log`, `week3_run.log`, `main_run.log` | Console captures from a local demo run (see §5). |

---

## 3. Mapping to the Final Project PDF (primary CO only)

| PDF expectation | Where it lives |
|-----------------|----------------|
| Understand the market; 5-minute OHLC | `CO.csv`; economics in `group5_config` + this README. |
| **Variance Ratio** and **Push-Response**; inference on inefficiency and time scales | `week1_timeseries_analysis.py` and run logs. |
| Implement **Channel WithDDControl**-style system | `main.py`: `rolling_hh_ll` + `run_backtest` (aligned with the course MATLAB port). |
| **ChnLen** 500–10_000 step **10**; **StpPct** 0.005–0.10 step **0.001** → **91 296** cells per full `optimize_parameters` | `group5_config.default_l_grid_pdf()`, `default_s_grid_pdf()`; used when `L_grid` / `S_grid` are `None`. |
| Optimize **net profit / max drawdown** (maximize ratio) | `optimize_window` / `optimize_parameters` use `return_to_dd`. |
| **~4 years** IS in bars, roll **~quarterly**, **~one quarter** OOS; save OOS equity, **optimal parameters each step**, performance and trade-style stats | `rolling_backtest`; `week3_rolling_oos.py` writes `oos_equity.csv`, `rolling_parameters.csv`, `oos_trades.csv`. |
| **Slpg**, **point value** consistent with the contract | Defaults in `group5_config`; **override** with TF Data via `group5_overrides.json`. |

**Not automated here but the PDF may still ask for it:** full secondary-market replication; systematic sweeps over IS length **T** and OOS length **τ**; full-sample IS-only optimization vs walk-forward decay comparison, etc.

---

## 4. How to run

### 4.1 Week 1 (full sample; usually a few minutes)

```bash
set MPLBACKEND=Agg
python week1_timeseries_analysis.py
```

Outputs: `week1_fig1_prices_returns.png`, `week1_fig2_variance_ratio.png`, `week1_fig3_push_response.png`, plus printed tables and interpretation.

### 4.2 Week 2 (parameter surface)

- **PDF full grid (951 × 96 = 91 296 evaluations per call; can take many hours):**

```bash
set MPLBACKEND=Agg
python week2_optimization.py
```

- **Development subset (fast; used for the demo numbers in §5–§8):**

```bash
set MPLBACKEND=Agg
python week2_optimization.py --fast-grid
```

### 4.3 Week 3 (rolling walk-forward OOS)

- **Default: PDF full grid inside every training window (very slow):**

```bash
set MPLBACKEND=Agg
python week3_rolling_oos.py --no-show --save-fig
```

- **Development: subset grid + cap the number of OOS windows:**

```bash
set MPLBACKEND=Agg
python week3_rolling_oos.py --fast-grid --max-segments 12 --no-show --save-fig
```

Writes: `oos_equity.csv`, `rolling_parameters.csv`, `oos_trades.csv`, `rolling_oos_equity_dd.png`.

### 4.4 `main.py` demo (fixed IS/OOS slice — **not** the Week 3 rolling experiment)

```bash
set MPLBACKEND=Agg
python main.py
```

---

## 5. What “test / cut-down” vs “formal” means

- **`--fast-grid`**: uses a **small** `(L,S)` subset so one `optimize_parameters` call finishes quickly. The syllabus, however, specifies the **full 91 296-point** grid for the optimization step.
- **`--max-segments 12`**: stops after **12** OOS quarters. The syllabus asks to roll **through all remaining quarters** on the full sample.

So: **yes — that combination is a test / truncated configuration**, not the formal submission configuration. **Finishing “for real” means long runs:** full grid on Week 2, and on Week 3 **each** training window must run the full grid, and you must **not** cap segments (unless the instructor approves an alternative).

**Which weeks / artifacts this touches**

| Flag | Affects | Data | Typical figures / CSVs |
|------|---------|------|-------------------------|
| `--fast-grid` | **Week 2** and **Week 3** (anything calling `optimize_parameters` with that override) | Still **`CO.csv`** | Week 2: `optimization_heatmap_*.png`, `optimization_results*.csv` — values differ from full grid. Week 3: `rolling_oos_equity_dd.png`, `oos_equity.csv`, `rolling_parameters.csv`, `oos_trades.csv` — **path and shape** are the same as production, **numbers** are for the truncated run only. |
| `--max-segments` | **Week 3 only** | Same **`CO.csv`** | Shorter stitched OOS curve and fewer rows in `rolling_parameters.csv` / `oos_trades.csv`. |
| *(none of the above)* | **Week 1** | **`CO.csv`** | `week1_fig*.png` — Week 1 in the demo run was **full-sample** exploratory analysis (no fast flag). |

---

## 6. Local demo run recorded in this README (not formal submission)

**Run date (reference):** 2026-04-23 (see file timestamps on your machine).

**Commands used for the logged demo:**

| Step | Command | Log file |
|------|---------|----------|
| Week 1 | `python week1_timeseries_analysis.py` | `week1_run.log` |
| Week 2 | `python week2_optimization.py --fast-grid` | `week2_run.log` |
| Week 3 | `python week3_rolling_oos.py --fast-grid --max-segments 12 --no-show --save-fig` | `week3_run.log` |
| `main.py` | `python main.py` | `main_run.log` |

The numbers in §7–§9 below come from that **truncated** configuration. **Before submitting**, rerun **without** `--fast-grid` and **without** `--max-segments`, and set `slpg` / `pv` from TF Data in `group5_overrides.json`.

`main.py` log snippet: in-sample vs out-of-sample summary line for a single `(L,S)` on a **fixed** date split — e.g. in/out blocks like `[44915, -13007, 38.6, 319.5]` vs `[69378, -6320, 55.1, 193.5]` (units as printed by `main.py`).

---

## 7. Week 1 — method and demo-run takeaways

### 7.1 What it does

1. Build a datetime index from `Date` + `Time`, clean OHLC, drop non-positive closes for finite log returns.  
2. Log returns: mean, std, skew, excess kurtosis.  
3. **Variance ratio** (overlapping) at `k ∈ {2,4,8,16,32}` vs an i.i.d. random walk.  
4. **Push-response**: “large” moves at the 95% quantile of \|r\|; conditional mean cumulative log return over horizons `1,5,10,20` bars after up vs down pushes.

### 7.2 Demo numbers (from `week1_run.log`)

- **n ≈ 384 305** log-return bars after cleaning.  
- **Heavy tails:** excess kurtosis very large (~572); skew negative (~-3.12).  
- **VR:** all tested `k` have **VR(k) < 1** (e.g. k=32: VR≈0.959) with **very small p-values** (min ~1.78e-7) → strong rejections of the simple i.i.d. RW null in the homoskedastic LM-style test used.  
- **Push-response:** small conditional mean magnitudes; asymmetry across up vs down pushes; see log tables.

### 7.3 Link to later weeks

Week 1 **does not** call the trading rules. It is **exploratory** context for whether a trend-style channel system is even plausible; it **does not replace** Week 3 OOS evidence.

---

## 8. Week 2 — method and demo-run takeaways (`--fast-grid`)

### 8.1 What it does

On the **full CO series** (after `bars_back` warm-up), each grid `(L,S)` runs `run_strategy`; results sorted by **return_to_dd**; writes `optimization_results.csv`, `optimization_results_full.csv`, and two heatmaps.

### 8.2 Demo best point (subset grid only)

Best in the **6×5** fast grid: **L = 500**, **S = 0.005** with very high in-sample `return_to_dd` (~41 in the demo log). Treat this as **illustrative** of the pipeline — **not** the PDF-mandated 91 296-point surface.

### 8.3 Formal vs demo

The syllabus expects **91 296** evaluations per `optimize_parameters`. **`--fast-grid` = 30** evaluations — development / README speed only.

---

## 9. Week 3 — method and demo-run takeaways (`--fast-grid`, 12 segments)

### 9.1 Logic (PDF-aligned structure)

1. Infer median bars per trading day → convert **4 years** and **3 months** to `is_bars`, `oos_bars`.  
2. Each roll: optimize on the last `is_bars` bars (grid as supplied — **full** unless `--fast-grid`).  
3. Run the strategy on a context long enough to start OOS at the split; stitch **OOS bar PnL** into `oos_equity.csv`; save per-window `(L,S)` and OOS metrics; export **completed round-turns** to `oos_trades.csv`.

### 9.2 Demo console summary (`week3_run.log`)

- **49 896** rows in `oos_equity.csv` (12 quarters stitched at 5-minute steps).  
- **12** rolling windows; `is_bars ≈ 66 528`, `oos_bars ≈ 4 158`, `bars_per_trading_day ≈ 66`.  
- Stitched OOS path (from $100k start): about **+33 558** global return vs about **-17 995** global max drawdown (see log).  
- **219** completed trades in `oos_trades.csv` for this truncated run.

### 9.3 Per-window table (abbreviated; full CSV is authoritative)

| segment | best_L | best_S | oos_return | oos_return_to_dd | oos_n_trades_roundturn | oos_win_rate_trades | oos_profit_factor |
|--------:|-------:|-------:|-------------:|-----------------:|----------------------:|--------------------:|------------------:|
| 0 | 500 | 0.02 | 3 280 | 0.60 | 12 | 0.25 | 0.78 |
| 1 | 500 | 0.01 | 4 064 | 1.01 | 27 | 0.44 | 1.26 |
| 2 | 500 | 0.01 | -5 267 | -0.52 | 32 | 0.25 | 0.45 |
| 3 | 1000 | 0.01 | 3 147 | 0.43 | 22 | 0.32 | 1.23 |
| 4 | 500 | 0.01 | 2 400 | 0.20 | 63 | 0.35 | 0.47 |
| 5 | 1000 | 0.03 | 4 516 | 0.45 | 21 | 0.29 | 1.01 |
| 6 | 1000 | 0.03 | 4 041 | 1.44 | 5 | 0.40 | 2.77 |
| 7 | 1000 | 0.03 | 4 726 | 0.95 | 10 | 0.40 | 1.41 |
| 8 | 1000 | 0.03 | 719 | 0.17 | 5 | 0.40 | 1.49 |
| 9 | 1000 | 0.03 | 4 881 | 0.74 | 5 | 0.60 | 2.27 |
| 10 | 1000 | 0.03 | 6 633 | 1.24 | 5 | 0.40 | 0.75 |
| 11 | 500 | 0.03 | 467 | 0.04 | 12 | 0.33 | 0.70 |

**Reading:** parameters **drift** across windows; some OOS windows have **weak or negative** `return_to_dd` — expected under walk-forward and useful to discuss **decay** and **stability** in the write-up.

### 9.4 End of stitched equity (`oos_equity.csv`)

Last demo row near **2010-07-13 16:30**: equity **≈ 133 558** from a **100 000** start, consistent with `global_return` in the log.

---

## 10. Contract economics (reminder)

- **`pv`**: default **1000** USD PnL per **$1/barrel** move for a 1000-barrel contract-style scaling — **verify** against TF Data.  
- **`slpg`**: default **80** USD round-turn is a **placeholder** — **replace** with TF Data column **V** for CO.  
- **Full-grid Week 3:** runtime ≈ (number of rolling windows) × (91 296 × cost of one backtest) — plan hardware and document runtime in the report.

---

## 11. Troubleshooting

| Issue | Fix |
|-------|-----|
| `No module named scipy` | `pip install -r requirements.txt`. |
| CSV not found | Run from the project root or fix `PRIMARY_DATA_FILE` in `group5_config.py`. |
| Week 2/3 too slow | Use `--fast-grid` (and optional `--max-segments`) for development; **remove** for formal runs. |
| Heatmaps mostly NaN | Often a grid cell with no trades or undefined drawdown metrics — inspect printed diagnostics. |

---

## 12. Author note

Quantitative sections §7–§9 are **tied to the demo commands in §5**. After a full-grid / full-history rerun, save new logs (e.g. `week2_run_full.log`, `week3_run_full.log`) and replace the tables and narrative numbers accordingly.

Simon
