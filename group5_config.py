"""
GR5360 Final Project — Group 5 primary market: CO (Brent crude), TD ticker CO.

PDF grids (ChnLen / StpPct) and contract economics. Slippage and point value must
match TF Data.xls / instructor; optional JSON overrides without editing code.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PRIMARY_DATA_FILE = "CO.csv"

# Secondary (grey-zone list in PDF): CME Bitcoin futures, TickData-style file in repo.
SECONDARY_DATA_FILE = "BTC-5minHLV.csv"

# ICE-style Brent: $/bbl in data × 1,000 bbl/contract → $ PnL per $1/bbl (verify TF Data column H).
DEFAULT_PV = 1000.0

# Full round-turn transaction cost ($) for CO from instructor table.
DEFAULT_SLPG = 48.0

# CME BTC (TD BTC): use instructor-provided economics for this project.
SECONDARY_DEFAULT_PV = 25.0
SECONDARY_DEFAULT_SLPG = 25.0

_OVERRIDES_PATH = Path(__file__).resolve().parent / "group5_overrides.json"


def contract_slippage_point_value(market: str = "primary") -> tuple[float, float]:
    """
    Return (slpg, pv) for backtests.

    Parameters
    ----------
    market:
        ``"primary"`` = CO (Brent). ``"secondary"`` = BTC file in ``SECONDARY_DATA_FILE``.

    ``group5_overrides.json`` (optional):
        Legacy top-level ``{\"slpg\", \"pv\"}`` applies to **primary** only.
        Optional nested ``{\"secondary\": {\"slpg\", \"pv\"}}`` for BTC.
    """
    m = "secondary" if str(market).lower() == "secondary" else "primary"
    if _OVERRIDES_PATH.is_file():
        data = json.loads(_OVERRIDES_PATH.read_text(encoding="utf-8"))
        if m == "secondary":
            sec = data.get("secondary")
            if isinstance(sec, dict):
                return float(sec.get("slpg", SECONDARY_DEFAULT_SLPG)), float(sec.get("pv", SECONDARY_DEFAULT_PV))
            return SECONDARY_DEFAULT_SLPG, SECONDARY_DEFAULT_PV
        prim = data.get("primary")
        if isinstance(prim, dict):
            return float(prim.get("slpg", DEFAULT_SLPG)), float(prim.get("pv", DEFAULT_PV))
        return float(data.get("slpg", DEFAULT_SLPG)), float(data.get("pv", DEFAULT_PV))
    if m == "secondary":
        return SECONDARY_DEFAULT_SLPG, SECONDARY_DEFAULT_PV
    return DEFAULT_SLPG, DEFAULT_PV


def default_l_grid_pdf() -> np.ndarray:
    """ChnLen: 500 … 10,000 step 10 (951 points)."""
    return np.arange(500, 10_000 + 1, 10, dtype=int)


def default_s_grid_pdf() -> np.ndarray:
    """StpPct: 0.005 … 0.100 step 0.001 (96 points)."""
    return (np.arange(5, 101, dtype=np.int64) / 1000.0).astype(float)
