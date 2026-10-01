"""Day-ahead power forecast: gradient-boosted ensembles + conformalized prediction intervals.

Products (hourly mean power, W)
-------------------------------
* ``pv``, ``wind``, ``total``             -- **delivered** power, as measured by the INA219s: what reaches the
                                             battery, including PV curtailment when the battery is full.
* ``pv_potential``, ``total_potential``   -- **generation potential**: what the panel / turbine can produce.
                                             Learned and scored only on hours without curtailment, the
                                             standard practice in grid PV forecasting. Wind is not curtailed
                                             (it is diverted to the dump load and still measured), so wind has one product.

Model per product
-----------------
* **Point**: an ensemble of ``ENSEMBLE_SEEDS`` LightGBMs (bagging with different seeds, averaged), early-stopped on
  the calibration days. PV is learned as a *clear-sky-normalised* target y / (GHI_clear / 1000), which removes
  the daily solar cycle from what the trees must learn; night PV is exactly zero from solar geometry.
* **Intervals**: conformalized quantile regression (CQR, Romano et al. 2019) per lead-time block (1-6, 7-24,
  25-48 h): quantile LightGBMs give a raw band, widened on the held-out calibration days by the (1-α)(1+1/n)
  quantile of the conformity score E = max(q_lo − y, y − q_hi), so the stated coverage holds on new data.

Choices were made on 2025 data only (train vs calibration days); see ``reports/forecast_evaluation.html``.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..config import Config
from .forecast_features import FEATURES

INTERVALS = {"95": (0.025, 0.975), "80": (0.10, 0.90)}
LEAD_BLOCKS = ((1, 6), (7, 24), (25, 48))
NIGHT_CLEAR_SKY_WM2 = 1.0
CSI_MIN_CLEAR_SKY = 20.0          # below this the clear-sky-normalised target is ill-conditioned (dawn/dusk)
ENSEMBLE_SEEDS = (0, 1, 2)
PRODUCTS = {"pv": ("pv", False), "wind": ("wind", False), "total": ("total", False),
            "pv_potential": ("pv", True), "total_potential": ("total", True)}
POINT_PARAMS = {"n_estimators": 3000, "learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 40,
                "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0}
QUANTILE_PARAMS = {"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 60,
                   "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8}
EARLY_STOP = 150
OPEN_METEO_URL = ("https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                  "&hourly=wind_speed_10m,shortwave_radiation,temperature_2m&wind_speed_unit=ms&timezone=UTC"
                  "&forecast_days=3")


def _block(lead: np.ndarray) -> np.ndarray:
    out = np.zeros(len(lead), dtype=int)
    for k, (a, b) in enumerate(LEAD_BLOCKS):
        out[(lead >= a) & (lead <= b)] = k
    return out


def not_curtailed(df: pd.DataFrame) -> np.ndarray:
    """Rows whose valid hour had no PV curtailment (controller never in float)."""
    return (df["y_float"].fillna(0.0) == 0.0).to_numpy()


@dataclass
class DayAheadModel:
    """Point ensemble + conformal intervals for one product."""

    product: str
    seed: int = 0
    members: list = field(default_factory=list)
    quantiles: dict = field(default_factory=dict)
    conformal: dict = field(default_factory=dict)        # (interval, lead block) -> widening [W]

    @property
    def target(self) -> str:
        return PRODUCTS[self.product][0]

    @property
    def potential(self) -> bool:
        return PRODUCTS[self.product][1]

    @property
    def y(self) -> str:
        return f"y_{self.target}"

    @property
    def csi(self) -> bool:
        return self.target == "pv"

    def _rows(self, d: pd.DataFrame) -> pd.DataFrame:
        d = d.dropna(subset=[self.y])
        return d[not_curtailed(d)] if self.potential else d

    def _night(self, X: pd.DataFrame) -> np.ndarray:
        return (X["ghi_clear"] <= NIGHT_CLEAR_SKY_WM2).to_numpy() if self.target == "pv" else np.zeros(len(X), bool)

    def _scale(self, X: pd.DataFrame) -> np.ndarray:
        return np.maximum(X["ghi_clear"].to_numpy(), CSI_MIN_CLEAR_SKY) / 1000.0 if self.csi else np.ones(len(X))

    def fit(self, train: pd.DataFrame, calib: pd.DataFrame) -> "DayAheadModel":
        tr, ca = self._rows(train), self._rows(calib)
        if self.csi:     # learn the clear-sky index; night rows carry no information
            tr = tr[tr["ghi_clear"] > NIGHT_CLEAR_SKY_WM2]
            ca_fit = ca[ca["ghi_clear"] > NIGHT_CLEAR_SKY_WM2]
        else:
            ca_fit = ca
        y_tr = tr[self.y].to_numpy() / self._scale(tr)
        y_ca = ca_fit[self.y].to_numpy() / self._scale(ca_fit)
        for s in ENSEMBLE_SEEDS:
            m = lgb.LGBMRegressor(random_state=self.seed + s, verbose=-1, **POINT_PARAMS)
            m.fit(tr[FEATURES], y_tr, eval_set=[(ca_fit[FEATURES], y_ca)], eval_metric="l1",
                  callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
            self.members.append(m)
        # intervals on the original scale (train-part quantile models, conformal widening on calibration days)
        tr_all, y_all = self._rows(train), self._rows(train)[self.y]
        for name, (lo, hi) in INTERVALS.items():
            qs = []
            for a in (lo, hi):
                q = lgb.LGBMRegressor(objective="quantile", alpha=a, random_state=self.seed, verbose=-1, **QUANTILE_PARAMS)
                qs.append(q.fit(tr_all[FEATURES], y_all))
            self.quantiles[name] = tuple(qs)
            raw_lo, raw_hi = qs[0].predict(ca[FEATURES]), qs[1].predict(ca[FEATURES])
            yc = ca[self.y].to_numpy()
            score = np.maximum(raw_lo - yc, yc - raw_hi)
            day = ~self._night(ca)
            blocks = _block(ca["lead_h"].to_numpy())
            alpha = 1.0 - (hi - lo)
            for k in range(len(LEAD_BLOCKS)):
                sc = score[(blocks == k) & day]
                n = len(sc)
                self.conformal[(name, k)] = float(np.quantile(sc, min(1.0, (1 - alpha) * (n + 1) / n))) if n else 0.0
        return self

    def predict_point(self, X: pd.DataFrame) -> np.ndarray:
        p = np.mean([m.predict(X[FEATURES], num_iteration=m.best_iteration_) for m in self.members], axis=0)
        p = np.maximum(p * self._scale(X), 0.0)
        p[self._night(X)] = 0.0
        return p

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        p = self.predict_point(X)
        night = self._night(X)
        out["p50"] = p
        blocks = _block(X["lead_h"].to_numpy())
        for name, (ql, qh) in self.quantiles.items():
            widen = np.array([self.conformal.get((name, k), 0.0) for k in blocks])
            lo = np.minimum(ql.predict(X[FEATURES]) - widen, p)       # measured power can sit a hair below 0
            hi = np.maximum(qh.predict(X[FEATURES]) + widen, p)
            lo[night] = hi[night] = 0.0
            out[f"lo{name}"], out[f"hi{name}"] = np.maximum(lo, 0.0), hi
        return out

    def importance(self, top: int = 12) -> pd.Series:
        imp = sum(pd.Series(m.booster_.feature_importance("gain"), index=FEATURES) for m in self.members)
        return (imp / imp.sum()).sort_values(ascending=False).head(top)


@dataclass
class DayAheadForecaster:
    """All products plus the monthly-hourly climatology used as a baseline."""

    models: dict = field(default_factory=dict)
    climatology: dict = field(default_factory=dict)

    def fit(self, train: pd.DataFrame, calib: pd.DataFrame, seed: int = 0, log=print) -> "DayAheadForecaster":
        for prod in PRODUCTS:
            self.models[prod] = DayAheadModel(prod, seed=seed).fit(train, calib)
            log(f"  {prod}: trees {[m.best_iteration_ for m in self.models[prod].members]}")
        both = pd.concat([train, calib])
        month = pd.DatetimeIndex(both["valid_time"]).month
        for t in ("pv", "wind", "total"):
            self.climatology[t] = both.groupby([month, both["local_hour"].to_numpy()])[f"y_{t}"].mean().to_dict()
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([m.predict(X).add_prefix(f"{prod}_") for prod, m in self.models.items()], axis=1)

    def climatology_forecast(self, X: pd.DataFrame, target: str) -> np.ndarray:
        month = pd.DatetimeIndex(X["valid_time"]).month
        return np.array([self.climatology[target].get((m, h), np.nan) for m, h in zip(month, X["local_hour"])])


def fetch_open_meteo(cfg: Config, timeout_s: float = 10.0) -> pd.DataFrame:
    """Live Open-Meteo hourly forecast for the site, in the schema of ``forecast_issued.parquet``."""
    loc = cfg.site["location"]
    url = OPEN_METEO_URL.format(lat=loc["latitude_deg"], lon=loc["longitude_deg"])
    with urllib.request.urlopen(url, timeout=timeout_s) as r:
        js = json.loads(r.read().decode())
    h = js["hourly"]
    issue = pd.Timestamp.now(tz="UTC").floor("h")
    fc = pd.DataFrame({"issue_time": issue, "valid_time": pd.to_datetime(h["time"]).tz_localize("UTC"),
                       "wind_speed_10m": h["wind_speed_10m"], "shortwave_radiation": h["shortwave_radiation"],
                       "temperature_2m": h["temperature_2m"]})
    fc["lead_h"] = ((fc["valid_time"] - issue) / pd.Timedelta(hours=1)).astype(int)
    return fc[(fc["lead_h"] >= 1) & (fc["lead_h"] <= 48)].reset_index(drop=True)
