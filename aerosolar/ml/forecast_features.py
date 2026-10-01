"""Hourly actuals and the feature table of the day-ahead power forecast.

Leakage rule (enforced here, tested in ``tests/test_forecast.py``): a row issued at time *i* may only use
(a) the weather forecast issued at *i* and (b) measurements from hours that **ended** by *i*.

Features per (issue time i, valid hour v, lead L = v - i)
-------------------------------------------------------
Weather forecast (issued at i):
    wind_speed_10m, shortwave_radiation, temperature_2m at v, at v-1 and v+1 (temporal context), and the
    mean wind / max and total radiation of v's local day (the "kind of day")
Solar geometry (deterministic):
    clear-sky GHI at v, forecast clear-sky index kt = shortwave / clear-sky, hour-of-day and day-of-year cycles
Observations up to i:
    last-hour PV power, wind power and measured wind; battery SOC and float (curtailment) share;
    PV yield over the last 72 h relative to clear-sky (tracks panel soiling and cleaning);
    measured / forecast wind ratio over the last 72 h (tracks the NWP roof-level bias);
    persistence: the actual at the same clock hour on the latest day already measured at i
Lead time L, and the cumulative forecast irradiance / wind from i to v (sw_cum, ws_cum): with the battery
SOC at i they tell the model whether the battery will be full by v, i.e. whether the charge controller will
*curtail* PV (10-11 % of PV energy is curtailed in spring and summer).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..sim.weather import clear_sky_ghi, solar_position
from .data_quality import clean
from .dataset import load_run_1hz, minute_table

MIN_VALID_MINUTES = 40
HORIZON_H = 48
CLEAR_SKY_SUBSTEPS = 12          # 5-minute grid for hourly-mean clear-sky GHI
OBS_WINDOW_H = 72
FLOAT_STATE = 2
TARGETS = ("pv", "wind", "total")
TABLE_VERSION = "2"           # bump when feature/target logic changes: cached tables are rebuilt

FEATURES = [
    "lead_h", "ws10", "ws10_prev", "ws10_next", "sw", "sw_prev", "sw_next", "t2m", "day_ws10_mean", "day_sw_max",
    "day_sw_sum", "ghi_clear", "kt_fc", "hod_sin", "hod_cos", "doy_sin", "doy_cos",
    "obs_pv_1h", "obs_wind_power_1h", "obs_wind_1h", "obs_soc", "obs_float_frac", "pv_yield_72h", "wind_bias_72h",
    "pers_pv", "pers_wind", "sw_cum", "ws_cum",
]


def clear_sky_hourly(hour_start_utc: pd.DatetimeIndex, cfg: Config) -> np.ndarray:
    """Hourly-mean clear-sky GHI [W/m^2] for hours starting at ``hour_start_utc``."""
    loc = cfg.site["location"]
    base = pd.DatetimeIndex(hour_start_utc).tz_convert("UTC").tz_localize(None).to_numpy().astype("datetime64[s]")
    offs = (np.arange(CLEAR_SKY_SUBSTEPS) * (3600 // CLEAR_SKY_SUBSTEPS) + 150).astype("timedelta64[s]")
    t = (base[:, None] + offs[None, :]).ravel()
    zen, _ = solar_position(t, loc["latitude_deg"], loc["longitude_deg"])
    return clear_sky_ghi(zen).reshape(len(base), CLEAR_SKY_SUBSTEPS).mean(axis=1)


def hourly_actuals(run: Path, cfg: Config) -> pd.DataFrame:
    """Cleaned hourly means of what the Pi measured (UTC hour start index). Hours with < 40 valid minutes are NaN."""
    df1, _ = clean(load_run_1hz(run))
    m = minute_table(df1, cfg).set_index("ts")
    g = m.resample("1h")
    h = pd.DataFrame({"wind_ms": g["wind_ms"].mean(), "p_wind": g["power_w"].mean(), "p_pv": g["pv_power_w"].mean(),
                      "soc": g["soc_pct"].mean(), "brake_frac": g["brake_frac"].mean(),
                      "float_frac": (m["chg_state"] == FLOAT_STATE).astype(float).resample("1h").mean(),
                      "n_wind": g["power_w"].count(), "n_pv": g["pv_power_w"].count()})
    h.loc[h["n_wind"] < MIN_VALID_MINUTES, ["p_wind", "wind_ms"]] = np.nan
    h.loc[h["n_pv"] < MIN_VALID_MINUTES, "p_pv"] = np.nan
    # generated power cannot be negative: small negative readings are the INA219 zero offset, not energy
    h[["p_wind", "p_pv"]] = h[["p_wind", "p_pv"]].clip(lower=0.0)
    h["p_total"] = h["p_wind"] + h["p_pv"]
    h["ghi_clear"] = clear_sky_hourly(h.index, cfg)
    return h.drop(columns=["n_wind", "n_pv"])


def _observations(hourly: pd.DataFrame, lead1: pd.Series) -> pd.DataFrame:
    """Observation features valid *at* each hour boundary i (uses hours that ended by i)."""
    h = hourly
    o = pd.DataFrame(index=h.index + pd.Timedelta(hours=1))       # hour [i-1, i) is known at i
    o["obs_pv_1h"] = h["p_pv"].to_numpy()
    o["obs_wind_power_1h"] = h["p_wind"].to_numpy()
    o["obs_wind_1h"] = h["wind_ms"].to_numpy()
    o["obs_soc"] = h["soc"].to_numpy()
    o["obs_float_frac"] = h["float_frac"].to_numpy()
    w = OBS_WINDOW_H
    day = h["ghi_clear"] > 50
    pv_sum = h["p_pv"].where(day).rolling(w, min_periods=12).sum()
    cs_sum = h["ghi_clear"].where(day & h["p_pv"].notna()).rolling(w, min_periods=12).sum()
    o["pv_yield_72h"] = (pv_sum / cs_sum.replace(0, np.nan)).to_numpy() * 1000.0
    fc1 = lead1.reindex(h.index)
    ok = h["wind_ms"].notna() & fc1.notna()
    o["wind_bias_72h"] = (h["wind_ms"].where(ok).rolling(w, min_periods=12).mean()
                          / fc1.where(ok).rolling(w, min_periods=12).mean()).to_numpy()
    return o


def build_table(forecasts: pd.DataFrame, hourly: pd.DataFrame, cfg: Config, horizon_h: int = HORIZON_H) -> pd.DataFrame:
    """One row per (issue, valid hour) with features and the measured targets (NaN where not measured)."""
    fc = forecasts[(forecasts["lead_h"] >= 1) & (forecasts["lead_h"] <= horizon_h)].copy()
    fc = fc.sort_values(["issue_time", "valid_time"]).reset_index(drop=True)
    g = fc.groupby("issue_time", sort=False)
    X = pd.DataFrame(index=fc.index)
    X["issue_time"], X["valid_time"], X["lead_h"] = fc["issue_time"], fc["valid_time"], fc["lead_h"]
    X["ws10"], X["sw"], X["t2m"] = fc["wind_speed_10m"], fc["shortwave_radiation"], fc["temperature_2m"]
    for c, src in (("ws10", "wind_speed_10m"), ("sw", "shortwave_radiation")):
        X[f"{c}_prev"] = g[src].shift(1).fillna(fc[src])
        X[f"{c}_next"] = g[src].shift(-1).fillna(fc[src])
    off = pd.Timedelta(hours=cfg.site["location"]["utc_offset_h"])
    vt_local = pd.DatetimeIndex(fc["valid_time"]).tz_convert(None) + off
    local_day = vt_local.normalize()
    key = [fc["issue_time"], local_day]
    X["day_ws10_mean"] = fc.groupby(key)["wind_speed_10m"].transform("mean").to_numpy()
    X["day_sw_max"] = fc.groupby(key)["shortwave_radiation"].transform("max").to_numpy()
    X["day_sw_sum"] = fc.groupby(key)["shortwave_radiation"].transform("sum").to_numpy()
    X["ghi_clear"] = clear_sky_hourly(pd.DatetimeIndex(fc["valid_time"]), cfg)
    X["kt_fc"] = np.clip(X["sw"] / np.maximum(X["ghi_clear"], 10.0), 0, 1.3).where(X["ghi_clear"] > 10, 0.0)
    hod = vt_local.hour + 0.5
    doy = vt_local.dayofyear
    X["hod_sin"], X["hod_cos"] = np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24)
    X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    # observations known at the issue time
    lead1 = fc[fc["lead_h"] == 1].set_index("valid_time")["wind_speed_10m"]
    obs = _observations(hourly, lead1)
    oi = obs.reindex(pd.DatetimeIndex(fc["issue_time"]))
    for c in obs.columns:
        X[c] = oi[c].to_numpy()
    # persistence: the same clock hour on the most recent day whose hour had *ended* by the issue time,
    # i.e. back = 24 h x ceil((L + 1) / 24)  (hour [v - back, v - back + 1) must end by i = v - L)
    vt = pd.DatetimeIndex(fc["valid_time"])
    back = 24 * np.ceil((fc["lead_h"].to_numpy() + 1) / 24.0).astype(int)
    prev_t = vt - pd.to_timedelta(back, unit="h")
    X["pers_pv"] = hourly["p_pv"].reindex(prev_t).to_numpy()
    X["pers_wind"] = hourly["p_wind"].reindex(prev_t).to_numpy()
    # targets
    a = hourly.reindex(vt)
    X["y_pv"], X["y_wind"], X["y_total"] = a["p_pv"].to_numpy(), a["p_wind"].to_numpy(), a["p_total"].to_numpy()
    X["local_hour"] = vt_local.hour
    X["local_date"] = local_day
    # share of the valid hour the controller spent in float (PV curtailed) -- target-side only, never a feature
    X["y_float"] = a["float_frac"].to_numpy()
    return add_derived(X)


def add_derived(X: pd.DataFrame) -> pd.DataFrame:
    """Cumulative forecast sums per issue (idempotent; also applied to cached tables)."""
    X = X.sort_values(["issue_time", "lead_h"])
    X["sw_cum"] = X.groupby("issue_time")["sw"].cumsum()
    X["ws_cum"] = X.groupby("issue_time")["ws10"].cumsum()
    return X


def run_table(run: Path, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(feature table, hourly actuals) for one generated run."""
    hourly = hourly_actuals(run, cfg)
    fc = pd.read_parquet(Path(run) / "forecast_issued.parquet")
    return build_table(fc, hourly, cfg), hourly
