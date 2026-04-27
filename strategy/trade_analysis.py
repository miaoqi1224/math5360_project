from __future__ import annotations

import numpy as np
import pandas as pd


def build_trade_level_table(
    market: pd.DataFrame,
    oos_trades: pd.DataFrame,
    *,
    price_column: str = "Close",
) -> pd.DataFrame:
    """
    Enrich OOS round-turn trades with entry/exit prices.

    Reuses week-3 output columns `entry_bar_global` and `exit_bar_global`.
    """
    if oos_trades.empty:
        cols = [
            "segment",
            "direction",
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "pnl",
            "holding_bars",
            "trade_return",
        ]
        return pd.DataFrame(columns=cols)

    if price_column not in market.columns:
        raise KeyError(f"Price column '{price_column}' not found in market data.")

    n = len(market)
    table = oos_trades.copy()
    entry_idx = table["entry_bar_global"].to_numpy(dtype=int)
    exit_idx = table["exit_bar_global"].to_numpy(dtype=int)
    if (entry_idx < 0).any() or (exit_idx < 0).any() or (entry_idx >= n).any() or (exit_idx >= n).any():
        raise IndexError("Trade global bar indices are out of market-data bounds.")

    prices = market[price_column].to_numpy(dtype=float)
    entry_prices = prices[entry_idx]
    exit_prices = prices[exit_idx]

    table["entry_price"] = entry_prices
    table["exit_price"] = exit_prices
    table["pnl"] = table["pnl_usd"].astype(float)
    table["direction"] = table["direction_closed"].astype(str)
    table["holding_bars"] = (table["exit_bar_global"] - table["entry_bar_global"]).astype(int)

    direction_sign = np.where(table["direction"].eq("long"), 1.0, -1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_ret = direction_sign * (table["exit_price"] - table["entry_price"]) / table["entry_price"]
    table["trade_return"] = raw_ret.replace([np.inf, -np.inf], np.nan)

    return table[
        [
            "segment",
            "direction",
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "pnl",
            "holding_bars",
            "trade_return",
        ]
    ].sort_values(["entry_time", "exit_time"], ignore_index=True)

