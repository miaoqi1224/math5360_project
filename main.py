"""
Python port of main.m - channel breakout backtest with trailing stop (StopPct).

Requires: pandas, numpy, matplotlib
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def load_market_data(data_file: str) -> pd.DataFrame:
    """Equivalent to ezread + numTime construction in MATLAB."""
    d = pd.read_csv(data_file)
    # Match MATLAB datenum-style ordering: Date + fractional time from Time column
    t_time = pd.to_datetime(d["Time"], format="%H:%M", errors="coerce")
    frac_day = (
        t_time.dt.hour + t_time.dt.minute / 60.0 + t_time.dt.second / 3600.0
    ) / 24.0
    d_date = pd.to_datetime(d["Date"], format="%m/%d/%Y", errors="coerce")
    d["numTime"] = mdates.date2num(d_date) + frac_day - np.floor(frac_day)
    d["N"] = len(d)
    d["M"] = 5
    return d


def datetime_from_mdy(s: str) -> pd.Timestamp:
    return pd.to_datetime(s, format="%m/%d/%Y")


def matlab_datenum_day_offset(date_str: str, day_offset: float = 0.0) -> float:
    """MATLAB datenum for mm/dd/yyyy plus integer day offset (e.g. end + 1)."""
    base = datetime_from_mdy(date_str)
    ts = base + pd.Timedelta(days=day_offset)
    return mdates.date2num(ts)


def rolling_hh_ll(high: np.ndarray, low: np.ndarray, L: int) -> tuple[np.ndarray, np.ndarray]:
    """HH/LL over previous L bars (current bar excluded), matching main.m."""
    hh = (
        pd.Series(high, dtype=float)
        .shift(1)
        .rolling(int(L), min_periods=int(L))
        .max()
        .to_numpy()
    )
    ll = (
        pd.Series(low, dtype=float)
        .shift(1)
        .rolling(int(L), min_periods=int(L))
        .min()
        .to_numpy()
    )
    return hh, ll


def run_backtest(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    hh: np.ndarray,
    ll: np.ndarray,
    S: float,
    bars_back: int,
    slpg: float,
    pv: float,
    e0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns equity E, drawdown DD, trades (per bar) matching main.m logic.
    Arrays are length n, 0-based indexing; meaningful trading from index bars_back onward.
    """
    n = len(close)
    E = np.zeros(n, dtype=float) + e0
    DD = np.zeros(n, dtype=float)
    trades = np.zeros(n, dtype=float)
    emax = float(e0)

    position = 0
    benchmark_long = np.nan
    benchmark_short = np.nan

    for k in range(bars_back, n):
        traded = False
        delta = pv * (close[k] - close[k - 1]) * position

        if position == 0:
            buy = high[k] >= hh[k]
            sell = low[k] <= ll[k]

            if buy and sell:
                delta = -slpg + pv * (ll[k] - hh[k])
                trades[k] = 1.0
            else:
                if buy:
                    delta = -slpg / 2.0 + pv * (close[k] - hh[k])
                    position = 1
                    traded = True
                    benchmark_long = high[k]
                    trades[k] = 0.5
                if sell:
                    delta = -slpg / 2.0 - pv * (close[k] - ll[k])
                    position = -1
                    traded = True
                    benchmark_short = low[k]
                    trades[k] = 0.5

        if position == 1 and not traded:
            sell_short = low[k] <= ll[k]
            sell = low[k] <= (benchmark_long * (1.0 - S))

            if sell_short and sell:
                if sell_short:
                    delta = delta - slpg - 2.0 * pv * (close[k] - ll[k])
                    position = -1
                    benchmark_short = low[k]
                    trades[k] = 1.0
            else:
                if sell:
                    delta = (
                        delta
                        - slpg / 2.0
                        - pv * (close[k] - (benchmark_long * (1.0 - S)))
                    )
                    position = 0
                    trades[k] = 0.5

                if sell_short:
                    delta = delta - slpg - 2.0 * pv * (close[k] - ll[k])
                    position = -1
                    benchmark_short = low[k]
                    trades[k] = 1.0

            benchmark_long = max(high[k], benchmark_long)

        if position == -1 and not traded:
            buy_long = high[k] >= hh[k]
            buy = high[k] >= (benchmark_short * (1.0 + S))

            if buy_long and buy:
                if buy_long:
                    delta = delta - slpg + 2.0 * pv * (close[k] - hh[k])
                    position = 1
                    benchmark_long = high[k]
                    trades[k] = 1.0
            else:
                if buy:
                    delta = (
                        delta
                        - slpg / 2.0
                        + pv * (close[k] - (benchmark_short * (1.0 + S)))
                    )
                    position = 0
                    trades[k] = 0.5

                if buy_long:
                    delta = delta - slpg + 2.0 * pv * (close[k] - hh[k])
                    position = 1
                    benchmark_long = high[k]
                    trades[k] = 1.0

            benchmark_short = min(low[k], benchmark_short)

        E[k] = E[k - 1] + delta
        emax = max(emax, E[k])
        DD[k] = E[k] - emax

    return E, DD, trades


# ---------------------------------------------------------------------------
# Week 2 - reusable wrapper + metrics + grid search (strategy logic unchanged)
# ---------------------------------------------------------------------------


def run_strategy(
    data: pd.DataFrame,
    L: int,
    S: float,
    *,
    bars_back: int = 17001,
    slpg: float = 47.0,
    pv: float = 42000.0,
    e0: float = 100_000.0,
) -> dict[str, np.ndarray]:
    """
    Run one backtest for channel length L and stop fraction S.

    Internally reuses rolling_hh_ll() and run_backtest() only (no duplicated
    trading rules). Does not rely on global variables.

    Returns
    -------
    dict with keys: "E", "DD", "trades", "pnl"
        pnl[t] is the bar-to-bar equity change (aligned with MATLAB-style
        construction: zeros before bars_back, then E[t]-E[t-1]).
    """
    high = data["High"].to_numpy(dtype=float)
    low = data["Low"].to_numpy(dtype=float)
    close = data["Close"].to_numpy(dtype=float)
    n = len(close)

    hh, ll = rolling_hh_ll(high, low, int(L))
    E, DD, trades = run_backtest(
        high, low, close, hh, ll, float(S), int(bars_back), float(slpg), float(pv), float(e0)
    )

    pnl = np.zeros(n, dtype=float)
    bb = int(bars_back)
    if bb < n:
        pnl[bb:] = E[bb:] - E[bb - 1 : n - 1]

    return {"E": E, "DD": DD, "trades": trades, "pnl": pnl}


def compute_metrics(
    E: np.ndarray,
    DD: np.ndarray,
    trades: np.ndarray,
    pnl: np.ndarray,
    *,
    bars_back: int = 17001,
    bars_per_year: float = 78.0 * 252.0,
    eps: float = 1e-12,
) -> dict[str, float]:
    """
    Performance metrics on the full equity path (same convention as exploratory backtest).

    - total_return: terminal equity minus initial (absolute P&L).
    - max_drawdown: min(DD), typically <= 0 (underwater vs running peak).
    - return_to_dd_ratio: total_return / abs(max_drawdown); NaN if no drawdown.
    - sharpe_ratio: annualized Sharpe on per-bar PnL after warm-up (mean/std * sqrt(bars/year)).
    - total_trades: sum of `trades` vector (matches original #trades accounting).
    - win_rate: share of post-warmup bars with strictly positive PnL (bar-level proxy).
    - avg_trade_return: total_return divided by total_trades (activity-weighted).
    """
    e0 = float(E[0])
    total_return = float(E[-1] - e0)

    max_drawdown = float(np.min(DD))

    if abs(max_drawdown) < eps:
        return_to_dd = float("nan")
    else:
        return_to_dd = total_return / abs(max_drawdown)

    active = pnl[int(bars_back) :]
    active = active[np.isfinite(active)]
    std_pnl = float(np.std(active, ddof=1)) if active.size > 1 else 0.0
    mean_pnl = float(np.mean(active)) if active.size else 0.0
    if std_pnl < eps:
        sharpe_ratio = float("nan")
    else:
        sharpe_ratio = float(mean_pnl / std_pnl * np.sqrt(bars_per_year))

    total_trades = float(np.sum(trades))

    if active.size:
        win_rate = float(np.mean(active > 0.0))
    else:
        win_rate = float("nan")

    if total_trades > eps:
        avg_trade_return = float(total_return / total_trades)
    else:
        avg_trade_return = float("nan")

    return {
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "return_to_dd_ratio": return_to_dd,
        "sharpe_ratio": sharpe_ratio,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_trade_return": avg_trade_return,
    }


def optimize_parameters(
    data: pd.DataFrame,
    *,
    L_grid: np.ndarray | None = None,
    S_grid: np.ndarray | None = None,
    bars_back: int = 17001,
    slpg: float = 47.0,
    pv: float = 42000.0,
    e0: float = 100_000.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Grid search over (L, S). Each cell calls run_strategy + compute_metrics.

    Default grids (Week 2 spec):
        L: 500 .. 10000 step 500
        S: 0.005 .. 0.05 step 0.005
    """
    if L_grid is None:
        L_grid = np.arange(500, 10_000 + 1, 500, dtype=int)
    if S_grid is None:
        S_grid = np.arange(0.005, 0.0501, 0.005, dtype=float)

    rows: list[dict[str, float | int]] = []
    nL, nS = len(L_grid), len(S_grid)

    for i, L in enumerate(L_grid):
        if verbose:
            print(f"Grid row {i + 1}/{nL}: L = {L}")
        for S in S_grid:
            out = run_strategy(
                data, int(L), float(S), bars_back=bars_back, slpg=slpg, pv=pv, e0=e0
            )
            m = compute_metrics(
                out["E"],
                out["DD"],
                out["trades"],
                out["pnl"],
                bars_back=bars_back,
            )
            rows.append(
                {
                    "L": int(L),
                    "S": float(S),
                    "return": m["total_return"],
                    "max_dd": m["max_drawdown"],
                    "return_to_dd": m["return_to_dd_ratio"],
                    "sharpe": m["sharpe_ratio"],
                    "trades": m["total_trades"],
                    "win_rate": m["win_rate"],
                    "avg_trade_return": m["avg_trade_return"],
                }
            )

    return pd.DataFrame(rows)


def plot_optimization_heatmaps(
    results: pd.DataFrame,
    *,
    save_prefix: str = "optimization",
) -> tuple[plt.Figure, plt.Figure]:
    """
    Matplotlib-only heatmaps (no seaborn). Expects `results` from optimize_parameters.
    """
    L_vals = np.sort(results["L"].unique())
    S_vals = np.sort(results["S"].unique())

    def pivot_metric(col: str) -> np.ndarray:
        mat = np.full((len(S_vals), len(L_vals)), np.nan, dtype=float)
        idx_L = {v: j for j, v in enumerate(L_vals)}
        idx_S = {v: i for i, v in enumerate(S_vals)}
        for _, row in results.iterrows():
            mat[idx_S[row["S"]], idx_L[row["L"]]] = row[col]
        return mat

    rdd = pivot_metric("return_to_dd")
    shp = pivot_metric("sharpe")

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    im1 = ax1.imshow(
        rdd,
        aspect="auto",
        origin="lower",
        extent=[L_vals[0], L_vals[-1], S_vals[0], S_vals[-1]],
        cmap="viridis",
    )
    ax1.set_xlabel("ChnLen L (bars)")
    ax1.set_ylabel("StopPct S")
    ax1.set_title("Heatmap: return / |max drawdown|")
    plt.colorbar(im1, ax=ax1, label="return_to_dd")
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    im2 = ax2.imshow(
        shp,
        aspect="auto",
        origin="lower",
        extent=[L_vals[0], L_vals[-1], S_vals[0], S_vals[-1]],
        cmap="magma",
    )
    ax2.set_xlabel("ChnLen L (bars)")
    ax2.set_ylabel("StopPct S")
    ax2.set_title("Heatmap: annualized Sharpe (per-bar PnL, post warm-up)")
    plt.colorbar(im2, ax=ax2, label="Sharpe")
    fig2.tight_layout()

    fig1.savefig(f"{save_prefix}_heatmap_return_to_dd.png", dpi=160)
    fig2.savefig(f"{save_prefix}_heatmap_sharpe.png", dpi=160)

    return fig1, fig2


def week2_interpretation_text(results: pd.DataFrame, best_row: pd.Series) -> str:
    """Plain-text discussion for a final report (English)."""
    rdd = results["return_to_dd"].to_numpy(dtype=float)
    finite = rdd[np.isfinite(rdd)]
    q25, q75 = (np.nanquantile(finite, 0.25), np.nanquantile(finite, 0.75)) if finite.size else (np.nan, np.nan)
    iqr = q75 - q25 if np.isfinite(q25) and np.isfinite(q75) else np.nan

    top5 = results.sort_values("return_to_dd", ascending=False, na_position="last").head(5)
    spread_L = int(top5["L"].max() - top5["L"].min())
    spread_S = float(top5["S"].max() - top5["S"].min())
    if spread_L == 0 and spread_S == 0:
        plateau_note = "Top five tie on the same (L,S) grid point (re-check for duplicate rows or numerical ties)."
    elif spread_L == 0:
        plateau_note = (
            "Top five share the same channel length L; sensitivity is concentrated along StopPct S on this grid."
        )
    elif spread_S == 0:
        plateau_note = "Top five share the same S; sensitivity is concentrated along L on this grid."
    else:
        plateau_note = (
            "Top five span both L and S, suggesting a broader high-performing neighborhood (still verify OOS)."
        )

    lines = [
        "WEEK 2 - PARAMETER SURFACE (STRUCTURED INTERPRETATION)",
        "",
        "1) Stability across parameters:",
        "   Inspect the return_to_dd heatmap: large contiguous high-value regions suggest",
        "   stable performance; if good performance appears only as isolated pixels,",
        "   results are likely fragile to estimation noise and implementation detail.",
        "",
        "2) Good regions vs sharp peaks:",
        f"   Across the grid, IQR of return_to_dd is roughly {iqr:.4g} (higher spread implies",
        "   more sensitivity). Among the top five (L,S) cells, spreads dL and dS summarize",
        f"   how dispersed the leaders are (here: dL={spread_L}, dS={spread_S:.4g}). {plateau_note}",
        "",
        "3) Overfitting risk:",
        "   In-sample grid optimization systematically rewards noise unless validated with",
        "   hold-out periods, perturbation tests, and stability checks. Sharp peaks with",
        "   mediocre neighbors are a classic overfitting signature.",
        "",
        "4) Robustness of the chosen parameters:",
        f"   Selected best: L={int(best_row['L'])}, S={float(best_row['S']):.4f}.",
        "   Robustness requires: (i) sensitivity is smooth near (L,S), (ii) performance is",
        "   not driven by a handful of dates, (iii) out-of-sample metrics track in-sample",
        "   rankings at least directionally.",
        "",
        "Bottom line: treat the heatmap as exploratory; promote only parameters that remain",
        "competitive under walk-forward or cross-validation splits.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Week 3 - Rolling walk-forward OOS (reuses optimize_parameters + run_strategy)
# ---------------------------------------------------------------------------


def infer_bars_per_trading_day(df: pd.DataFrame) -> float:
    """
    Median number of 5-minute rows per calendar day (for converting years/months to bars).

    If parsing fails, falls back to ~78 bars/day (common RTH 5-min count heuristic).
    """
    dt = pd.to_datetime(
        df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip(),
        format="%m/%d/%Y %H:%M",
        errors="coerce",
    )
    day = dt.dt.normalize()
    counts = df.groupby(day, sort=False).size()
    med = float(counts.median())
    if not np.isfinite(med) or med < 1.0:
        return 78.0
    return med


def years_months_to_bars(
    df: pd.DataFrame,
    *,
    is_years: float = 4.0,
    oos_months: float = 3.0,
    trading_days_per_year: int = 252,
) -> tuple[int, int, float]:
    """
    Convert in-sample years and OOS months to bar counts using inferred bars/trading-day.

    Convention: ``is_bars ≈ round(is_years * 252 * bpd)``, ``oos_bars ≈ round((oos_months/12)*252*bpd)``.
    """
    bpd = infer_bars_per_trading_day(df)
    is_bars = int(max(1, round(float(is_years) * trading_days_per_year * bpd)))
    oos_bars = int(max(1, round((float(oos_months) / 12.0) * trading_days_per_year * bpd)))
    return is_bars, oos_bars, bpd


def _row_datetimes(df: pd.DataFrame) -> pd.Series:
    """One timestamp per row (for CSV / plots), aligned with iloc positions."""
    return pd.to_datetime(
        df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip(),
        format="%m/%d/%Y %H:%M",
        errors="coerce",
    ).reset_index(drop=True)


def optimize_window(
    train: pd.DataFrame,
    *,
    L_grid: np.ndarray | None = None,
    S_grid: np.ndarray | None = None,
    bars_back: int = 17001,
    slpg: float = 47.0,
    pv: float = 42000.0,
    e0: float = 100_000.0,
) -> tuple[int, float, pd.DataFrame]:
    """
    Run ``optimize_parameters`` on the in-sample slice only; pick (L, S) by max return_to_dd.

    Returns (best_L, best_S, full_grid_table).
    """
    grid = optimize_parameters(
        train,
        L_grid=L_grid,
        S_grid=S_grid,
        bars_back=bars_back,
        slpg=slpg,
        pv=pv,
        e0=e0,
        verbose=False,
    )
    if grid.empty:
        raise ValueError("optimize_parameters returned an empty grid.")
    scores = grid["return_to_dd"].astype(float).replace([np.inf, -np.inf], np.nan)
    if scores.notna().any():
        idx = scores.fillna(-np.inf).idxmax()
    else:
        idx = grid.index[0]
    best = grid.loc[idx]
    return int(best["L"]), float(best["S"]), grid


def rolling_backtest(
    data: pd.DataFrame,
    *,
    is_years: float = 4.0,
    oos_months: float = 3.0,
    bars_back: int = 17001,
    slpg: float = 47.0,
    pv: float = 42000.0,
    e0: float = 100_000.0,
    L_grid: np.ndarray | None = None,
    S_grid: np.ndarray | None = None,
    trading_days_per_year: int = 252,
    max_segments: int | None = None,
) -> dict[str, object]:
    """
    Rolling walk-forward OOS backtest (no look-ahead).

    For each step ``split`` (end of IS / start of OOS):

    1. **Train** = rows ``[split - is_bars, split)`` — only this slice is passed to ``optimize_window``.
    2. **OOS context** = rows ``[split - bars_back, split + oos_bars)`` — length ``bars_back + oos_bars`` so
       ``run_backtest`` can start trading exactly at global index ``split`` (local index ``bars_back``),
       using only prices **strictly before** each simulated bar inside OOS (standard causal HH/LL).
    3. Record OOS segment equity by **chaining dollar PnL** across all OOS bars in order.

    Walk-forward: after each OOS block, ``split += oos_bars`` (next IS ends at new split).

    Returns dict with ``oos_equity`` (DataFrame), ``rolling_parameters`` (DataFrame),
    ``global_return``, ``global_max_drawdown``, and diagnostic bar counts.
    """
    df = data.copy()
    ok = df["Close"].to_numpy(dtype=float) > 0.0
    df = df.loc[ok].reset_index(drop=True)
    n = len(df)
    is_bars, oos_bars, bpd = years_months_to_bars(
        df,
        is_years=is_years,
        oos_months=oos_months,
        trading_days_per_year=trading_days_per_year,
    )
    bb = int(bars_back)
    if is_bars <= bb:
        raise ValueError(
            f"In-sample window ({is_bars} bars) must exceed warm-up bars_back={bb}."
        )

    times = _row_datetimes(df)
    bars_per_year_ann = float(trading_days_per_year) * bpd

    rows: list[dict[str, object]] = []
    oos_times: list[pd.Timestamp] = []
    oos_equity: list[float] = []

    split = is_bars
    seg = 0
    while split + oos_bars <= n:
        is0 = split - is_bars
        train = df.iloc[is0:split]
        L_best, S_best, _ = optimize_window(
            train,
            L_grid=L_grid,
            S_grid=S_grid,
            bars_back=bb,
            slpg=slpg,
            pv=pv,
            e0=e0,
        )

        ctx0 = split - bb
        ctx1 = split + oos_bars
        ctx = df.iloc[ctx0:ctx1]
        out = run_strategy(
            ctx, L_best, S_best, bars_back=bb, slpg=slpg, pv=pv, e0=e0
        )
        E_loc = out["E"]
        DD_loc = out["DD"]
        pnl_loc = out["pnl"]
        trades_loc = out["trades"]

        lo = bb
        hi = bb + oos_bars
        E_oos = np.asarray(E_loc[lo:hi], dtype=float)
        pnl_oos = pnl_loc[lo:hi]
        trades_oos = trades_loc[lo:hi]
        # Segment-local drawdown (peak within OOS only); avoids mixing pre-OOS peaks from ctx.
        peak_oos = np.maximum.accumulate(E_oos)
        DD_oos = E_oos - peak_oos

        m_oos = compute_metrics(
            E_oos,
            DD_oos,
            trades_oos,
            pnl_oos,
            bars_back=0,
            bars_per_year=bars_per_year_ann,
        )

        rows.append(
            {
                "segment": seg,
                "train_start": times.iloc[is0],
                "train_end": times.iloc[split - 1],
                "oos_start": times.iloc[split],
                "oos_end": times.iloc[split + oos_bars - 1],
                "best_L": L_best,
                "best_S": S_best,
                "oos_return": m_oos["total_return"],
                "oos_max_dd": m_oos["max_drawdown"],
                "oos_return_to_dd": m_oos["return_to_dd_ratio"],
                "oos_sharpe": m_oos["sharpe_ratio"],
                "oos_trades": m_oos["total_trades"],
            }
        )

        for i in range(oos_bars):
            oos_times.append(times.iloc[split + i])
            prev = float(oos_equity[-1]) if oos_equity else float(e0)
            oos_equity.append(prev + float(pnl_oos[i]))

        split += oos_bars
        seg += 1
        if max_segments is not None and seg >= int(max_segments):
            break

    if not oos_equity:
        oos_df = pd.DataFrame(columns=["datetime", "equity", "drawdown"])
        dd_global = pd.Series(dtype=float)
    else:
        idx = pd.DatetimeIndex(oos_times)
        equity_s = pd.Series(oos_equity, index=idx, name="equity")
        peak = equity_s.cummax()
        dd_global = equity_s - peak
        oos_df = pd.DataFrame(
            {"datetime": equity_s.index, "equity": equity_s.values, "drawdown": dd_global.values}
        )

    params_df = pd.DataFrame(rows)
    g_ret = float(oos_equity[-1] - e0) if oos_equity else 0.0
    g_dd = float(dd_global.min()) if len(dd_global) else 0.0

    return {
        "oos_equity": oos_df,
        "rolling_parameters": params_df,
        "global_return": g_ret,
        "global_max_drawdown": g_dd,
        "bars_per_trading_day": bpd,
        "is_bars": is_bars,
        "oos_bars": oos_bars,
        "n_segments": int(seg),
    }


def plot_rolling_oos(
    oos_equity: pd.DataFrame,
    *,
    title_prefix: str = "Rolling OOS",
    figsize: tuple[float, float] = (11, 7),
) -> plt.Figure:
    """Equity and drawdown for the stitched walk-forward OOS path."""
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    ax0 = axes[0]
    ax1 = axes[1]
    if len(oos_equity) == 0:
        ax0.set_title(f"{title_prefix} (no OOS bars)")
        return fig
    ax0.plot(oos_equity["datetime"], oos_equity["equity"], color="tab:blue", linewidth=0.8)
    ax0.set_ylabel("Equity")
    ax0.set_title(f"{title_prefix} - stitched OOS equity")
    ax0.grid(True, alpha=0.25)
    ax1.plot(oos_equity["datetime"], oos_equity["drawdown"], color="tab:red", linewidth=0.8)
    ax1.set_ylabel("Drawdown")
    ax1.set_xlabel("Time")
    ax1.set_title("Drawdown (vs running peak)")
    ax1.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def main() -> None:
    data_file = "HO-5minHLV.csv"
    bars_back = 17001
    slpg = 47.0
    pv = 42000.0
    e0 = 100000.0

    in_sample = (datetime_from_mdy("01/01/1980"), datetime_from_mdy("01/01/2000"))
    out_sample = (datetime_from_mdy("01/01/2000"), datetime_from_mdy("03/23/2023"))

    length_grid = np.arange(12700, 12701, 100, dtype=int)
    stop_pct_grid = np.arange(0.010, 0.011, 0.001, dtype=float)

    result_label = ["Profit", "WorstDrawDown", "StDev", "#trades"]
    result_in_sample = np.zeros(
        (len(length_grid), len(stop_pct_grid), len(result_label))
    )
    result_out_sample = np.zeros_like(result_in_sample)

    d = load_market_data(data_file)
    num_time = d["numTime"].to_numpy(dtype=float)
    n = len(num_time)
    high = d["High"].to_numpy(dtype=float)
    low = d["Low"].to_numpy(dtype=float)
    close = d["Close"].to_numpy(dtype=float)

    # Figure 1 - full close series
    plt.figure(1)
    plt.clf()
    plt.plot(num_time, close, "b")
    plt.gca().xaxis_date()
    plt.title("Close (full sample)")
    plt.tight_layout()

    def count_before(ts: pd.Timestamp) -> int:
        return int(np.sum(num_time < mdates.date2num(ts)))

    # MATLAB: max(sum(d.numTime<inSample(1))+1, barsBack) — 1-based
    ind_in_sample_1_m = max(count_before(in_sample[0]) + 1, bars_back)
    ind_in_sample_2_m = max(
        int(np.sum(num_time < matlab_datenum_day_offset("01/01/2000", 1.0))), bars_back
    )
    ind_out_sample_1_m = max(count_before(out_sample[0]) + 1, bars_back)
    ind_out_sample_2_m = max(
        int(np.sum(num_time < matlab_datenum_day_offset("03/23/2023", 1.0))), bars_back
    )

    ind_in_sample_1 = ind_in_sample_1_m - 1
    ind_in_sample_2 = ind_in_sample_2_m - 1
    ind_out_sample_1 = ind_out_sample_1_m - 1
    ind_out_sample_2 = ind_out_sample_2_m - 1

    for i, L in enumerate(length_grid):
        print(f"calculating for Length = {L}")
        hh_L, ll_L = rolling_hh_ll(high, low, int(L))

        for j, S in enumerate(stop_pct_grid):
            E, DD, trades = run_backtest(
                high, low, close, hh_L, ll_L, float(S), bars_back, slpg, pv, e0
            )

            pnl = np.zeros(n)
            pnl[bars_back:] = E[bars_back:] - E[bars_back - 1 : n - 1]

            result_in_sample[i, j, :] = np.array(
                [
                    E[ind_in_sample_2] - E[ind_in_sample_1],
                    np.min(DD[ind_in_sample_1 : ind_in_sample_2 + 1]),
                    float(np.std(pnl[ind_in_sample_1 : ind_in_sample_2 + 1], ddof=0)),
                    np.sum(trades[ind_in_sample_1 : ind_in_sample_2 + 1]),
                ]
            )
            result_out_sample[i, j, :] = np.array(
                [
                    E[ind_out_sample_2] - E[ind_out_sample_1],
                    np.min(DD[ind_out_sample_1 : ind_out_sample_2 + 1]),
                    float(np.std(pnl[ind_out_sample_1 : ind_out_sample_2 + 1], ddof=0)),
                    np.sum(trades[ind_out_sample_1 : ind_out_sample_2 + 1]),
                ]
            )

            print(
                f"{S}: in/out: {result_in_sample[i, j, :].tolist()} / "
                f"{result_out_sample[i, j, :].tolist()}"
            )

    # Last grid S for plotting (matches MATLAB loop end)
    S_plot = float(stop_pct_grid[-1])
    L_plot = int(length_grid[-1])
    hh, ll = rolling_hh_ll(high, low, L_plot)
    E_out, _, trades_out = run_backtest(
        high, low, close, hh, ll, S_plot, bars_back, slpg, pv, e0
    )

    ind = slice(ind_out_sample_1, ind_out_sample_2 + 1)
    hh_i = hh[ind]
    ll_i = ll[ind]
    tr_i = trades_out[ind]

    def safe_half_over_tr(t: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(t != 0, 0.5 / t, np.nan)

    r = safe_half_over_tr(tr_i)

    plt.figure(2)
    plt.clf()
    plt.plot(hh_i, "g", label="HH")
    plt.plot(hh_i * (1.0 - S_plot), "--g")
    plt.plot(ll_i, "c", label="LL")
    plt.plot(ll_i * (1.0 + S_plot), "--c")
    plt.plot(r * hh_i, ".r")
    plt.plot(r * ll_i, ".r")
    plt.plot(r * hh_i * (1.0 - S_plot), ".r")
    plt.plot(r * ll_i * (1.0 + S_plot), ".r")
    plt.plot(tr_i * 4.0 + 2.0, "--r")
    plt.title("Out-of-sample window (numeric index)")
    plt.tight_layout()

    T = mdates.num2date(num_time[ind], tz=None)

    plt.figure(3)
    plt.clf()
    plt.plot(T, hh_i, "g")
    plt.plot(T, hh_i * (1.0 - S_plot), "--g")
    plt.plot(T, ll_i, "c")
    plt.plot(T, ll_i * (1.0 + S_plot), "--c")
    plt.plot(T, r * hh_i, ".r")
    plt.plot(T, r * ll_i, ".r")
    plt.plot(T, r * hh_i * (1.0 - S_plot), ".r")
    plt.plot(T, r * ll_i * (1.0 + S_plot), ".r")
    plt.plot(T, tr_i * 4.0 + 2.0, "--r")
    plt.gcf().autofmt_xdate()
    plt.title("Out-of-sample window (datetime)")
    plt.tight_layout()

    plt.figure(4)
    plt.clf()
    plt.plot(E_out[ind], "g")
    plt.title("Equity (out-of-sample)")
    plt.tight_layout()

    if plt.matplotlib.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
