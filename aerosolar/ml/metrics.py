"""Forecast accuracy statistics -- every one explicitly defined, because "accuracy" is ambiguous.

Notation: y actual, ŷ forecast, C the capacity of the forecast quantity [W].

* MAE, RMSE                         [W]
* R²                                1 − Σ(y−ŷ)² / Σ(y−ȳ)²
* capacity accuracy (RMSE)          1 − RMSE / C           (grid-code style, e.g. China GB/T 19964 / NB/T 32011)
* capacity accuracy (MAE)           1 − MAE / C            (= 1 − nMAE)
* daytime accuracy                  1 − Σ|y−ŷ| / Σy  over hours with y ≥ 10 % C (weighted MAPE, excludes nights)
* daily energy accuracy             mean over days of 1 − |Σŷ − Σy| / Σy  (energy per local day)
* skill vs baseline                 1 − MAE_model / MAE_baseline
* interval coverage                 share of actuals inside the stated P2.5–P97.5 (or P10–P90) band

Capacity-normalised metrics include night hours, where solar power is trivially zero; that is how grid codes
define them, but it flatters solar. The daytime and R² figures do not have that bias -- read them together.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAYTIME_FRACTION_OF_CAP = 0.10


def point_metrics(y: np.ndarray, p: np.ndarray, cap: float) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    err = p - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    day = y >= DAYTIME_FRACTION_OF_CAP * cap
    return {"n": int(ok.sum()), "mae_w": mae, "rmse_w": rmse, "bias_w": float(err.mean()),
            "r2": 1.0 - float(np.sum(err ** 2)) / ss_tot if ss_tot > 0 else np.nan,
            "cap_accuracy_rmse": 1.0 - rmse / cap, "cap_accuracy_mae": 1.0 - mae / cap,
            "daytime_accuracy": 1.0 - float(np.sum(np.abs(err[day])) / np.sum(y[day])) if day.any() else np.nan}


def daily_energy_accuracy(frame: pd.DataFrame, y: str, p: str, date_col: str = "local_date",
                          min_hours: int = 22) -> dict:
    """Per-day energy accuracy for complete days (≥ ``min_hours`` valid hours)."""
    d = frame[[date_col, y, p]].dropna()
    g = d.groupby(date_col)
    full = g.size() >= min_hours
    e_y = g[y].sum()[full]
    e_p = g[p].sum()[full]
    ok = e_y > 0
    acc = 1.0 - (e_p[ok] - e_y[ok]).abs() / e_y[ok]
    return {"days": int(ok.sum()), "daily_energy_accuracy_mean": float(acc.mean()) if len(acc) else np.nan,
            "daily_energy_accuracy_p10": float(acc.quantile(0.1)) if len(acc) else np.nan,
            "daily_energy_wh_mae": float((e_p[ok] - e_y[ok]).abs().mean()) if len(acc) else np.nan,
            "daily_energy_wh_mean": float(e_y[ok].mean()) if len(acc) else np.nan}


def interval_metrics(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict:
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    inside = (y[ok] >= lo[ok] - 1e-9) & (y[ok] <= hi[ok] + 1e-9)
    return {"coverage": float(inside.mean()), "mean_width_w": float(np.mean(hi[ok] - lo[ok]))}


def skill(mae_model: float, mae_baseline: float) -> float:
    return 1.0 - mae_model / mae_baseline if mae_baseline > 0 else np.nan
