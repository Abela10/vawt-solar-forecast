"""Data cleaning of the 1 Hz telemetry before it is used as forecast inputs or training targets.

A forecast model trained on corrupted actuals learns the corruption. Rules (per second, explainable):

    RANGE   outside physical limits
    GLITCH  DS18B20 codes 85.0 (power-on) and -127 (disconnect)
    STUCK   exactly constant for W seconds while a physically related signal moves
            (a flat ~zero is physical for power / current / lux at night or below cut-in; a BH1750 pinned at
            65535 lx is saturation, not a stuck sensor)
    ROC     rate of change faster than the physics allows

Flagged samples become NaN; the hourly actuals then require a minimum number of valid minutes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RANGE, GLITCH, STUCK, ROC = 1, 2, 4, 8
FLAG_NAMES = {RANGE: "range", GLITCH: "glitch", STUCK: "stuck", ROC: "rate"}
DS_FIELDS = ("t_gen_c", "t_bearing_c", "t_amb_c")
DS_CODES = (85.0, -127.0)

# field: (low, high, max |Δ| per s)
LIMITS = {
    "wind_ms": (0.0, 40.0, 20.0), "rpm": (0.0, 450.0, 400.0), "hall_count": (0.0, 8.0, 8.0),
    "bus_v": (0.0, 32.0, 8.0), "current_a": (-0.2, 3.3, 3.0), "power_w": (-1.0, 80.0, 60.0),
    "pv_power_w": (-0.5, 20.0, 15.0), "batt_v": (9.0, 15.5, 1.5), "soc_pct": (0.0, 100.0, 5.0),
    "t_gen_c": (-10.0, 110.0, 3.0), "t_bearing_c": (-10.0, 110.0, 3.0), "t_amb_c": (-5.0, 60.0, 3.0),
    "lux": (0.0, 65535.0, 65535.0), "ldr_ew": (-1.5, 1.5, 1.5), "ldr_ns": (-1.5, 1.5, 1.5),
    "dht_t_c": (-5.0, 60.0, 3.0), "dht_rh": (0.0, 100.0, 20.0),
}
# field: (window s, related field, min range of the related field inside the window)
STUCK_RULES = {
    "wind_ms": (300, "rpm", 30.0), "rpm": (300, "wind_ms", 2.0), "t_amb_c": (1800, "dht_t_c", 1.0),
    "dht_t_c": (900, "t_amb_c", 1.0), "power_w": (600, "rpm", 40.0), "pv_power_w": (600, "lux", 5000.0),
    "lux": (600, "pv_power_w", 2.0), "batt_v": (1800, "pv_power_w", 3.0),
}
STUCK_ZERO_OK = {"rpm": 0.5, "current_a": 0.01, "power_w": 0.1, "pv_power_w": 0.1, "lux": 5.0}
SATURATION = {"lux": 65535.0}


def rule_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Per-second rule flags (bitmask per field) for the 1 Hz Pi-side frame."""
    flags = pd.DataFrame(0, index=df.index, columns=[c for c in LIMITS if c in df.columns], dtype=np.int16)
    for c, (lo, hi, roc) in LIMITS.items():
        if c not in df:
            continue
        x = df[c]
        glitch = x.isin(DS_CODES) if c in DS_FIELDS else pd.Series(False, index=df.index)
        flags.loc[glitch, c] |= GLITCH
        xr = x.where(~glitch)
        flags.loc[(xr < lo) | (xr > hi), c] |= RANGE
        flags.loc[xr.diff().abs() > roc, c] |= ROC
    for c, (w, drv, thr) in STUCK_RULES.items():
        if c not in df or drv not in df:
            continue
        x = df[c].where(~df[c].isin(DS_CODES))
        r = x.rolling(w, min_periods=int(0.8 * w))
        flat = (r.max() - r.min()) == 0
        if c in STUCK_ZERO_OK:
            flat &= r.max().abs() > STUCK_ZERO_OK[c]
        if c in SATURATION:
            flat &= r.max() < SATURATION[c]
        dr = df[drv].rolling(w, min_periods=int(0.5 * w))
        flags.loc[flat & ((dr.max() - dr.min()) > thr), c] |= STUCK
    return flags


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (cleaned frame with flagged samples set to NaN, flags)."""
    flags = rule_flags(df)
    out = df.copy()
    for c in flags.columns:
        out.loc[flags[c] != 0, c] = np.nan
    if "rpm" in flags and "hall_count" in out:
        out.loc[flags["rpm"] != 0, "hall_count"] = np.nan
    return out, flags
