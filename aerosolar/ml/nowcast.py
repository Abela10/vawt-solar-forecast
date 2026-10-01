"""5-minute power nowcast from recent history.

At minute m the model predicts the mean PV, wind and total power over minutes m+1 … m+5 from the last
15 minutes of cleaned 1-minute means, plus the *known future* clear-sky GHI (solar geometry is
deterministic, so the sunset ramp is no surprise).

Baselines
* persistence           last minute's value
* smart persistence     (PV only) last 15 min clear-sky index × clear-sky GHI 5 min ahead, the standard solar
                        nowcast baseline -- it already knows the sun is setting
"""
from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

HORIZON_MIN = 5
LAGS = 15
SMART_WINDOW_MIN = 15
TARGETS = {"pv": "pv_power_w", "wind": "power_w"}
PARAMS = {"n_estimators": 800, "learning_rate": 0.04, "num_leaves": 63, "min_child_samples": 80,
          "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8}


def nowcast_table(m: pd.DataFrame) -> pd.DataFrame:
    """Features at each minute and the mean of the next ``HORIZON_MIN`` minutes as targets."""
    X = pd.DataFrame(index=m.index)
    for c in ("wind_ms", "power_w", "pv_power_w", "irr"):
        for k in range(LAGS):
            X[f"{c}_l{k}"] = m[c].shift(k)
        X[f"{c}_mean15"] = m[c].rolling(15, min_periods=5).mean()
        X[f"{c}_std15"] = m[c].rolling(15, min_periods=5).std()
    X["wind_std"] = m["wind_std"]
    X["wind_mean60"] = m["wind_ms"].rolling(60, min_periods=20).mean()
    X["ghi_clear"] = m["ghi_clear"]
    future_cs = pd.concat([m["ghi_clear"].shift(-k) for k in range(1, HORIZON_MIN + 1)], axis=1).mean(axis=1)
    X["ghi_clear_next"] = future_cs
    csi = (m["pv_power_w"].rolling(SMART_WINDOW_MIN, min_periods=5).mean()
           / m["ghi_clear"].rolling(SMART_WINDOW_MIN, min_periods=5).mean().where(lambda s: s > 20))
    X["pv_csi15"] = csi
    X["hod_sin"], X["hod_cos"] = m["hod_sin"], m["hod_cos"]
    X["brake_frac"], X["chg_state"], X["soc_pct"] = m["brake_frac"], m["chg_state"], m["soc_pct"]
    for t, c in TARGETS.items():
        fut = pd.concat([m[c].shift(-k) for k in range(1, HORIZON_MIN + 1)], axis=1)
        X[f"y_{t}"] = fut.mean(axis=1, skipna=False)
        X[f"persist_{t}"] = m[c]
    X["y_total"] = X["y_pv"] + X["y_wind"]
    X["persist_total"] = X["persist_pv"] + X["persist_wind"]
    X["smart_pv"] = (csi * future_cs).where(future_cs > 20, 0.0).fillna(X["persist_pv"])
    X["smart_total"] = X["smart_pv"] + X["persist_wind"]
    X["ts"] = m["ts"]
    return X


@dataclass
class Nowcaster:
    seed: int = 0
    models: dict = field(default_factory=dict)
    features: list = field(default_factory=list)

    def fit(self, table: pd.DataFrame, valid: pd.DataFrame) -> "Nowcaster":
        self.features = [c for c in table.columns
                         if not c.startswith(("y_", "persist", "smart", "ts")) and c != "run"]
        for t in TARGETS:
            d = table.dropna(subset=[f"y_{t}"])
            v = valid.dropna(subset=[f"y_{t}"])
            m = lgb.LGBMRegressor(random_state=self.seed, verbose=-1, **PARAMS)
            m.fit(d[self.features], d[f"y_{t}"], eval_set=[(v[self.features], v[f"y_{t}"])], eval_metric="l1",
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            self.models[t] = m
        return self

    def predict(self, table: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=table.index)
        for t, m in self.models.items():
            p = np.maximum(m.predict(table[self.features], num_iteration=m.best_iteration_), 0.0)
            if t == "pv":
                p = np.where(table["ghi_clear_next"] <= 1.0, 0.0, p)
            out[f"nowcast_{t}"] = p
        out["nowcast_total"] = out["nowcast_pv"] + out["nowcast_wind"]
        return out
