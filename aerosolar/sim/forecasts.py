"""Synthetic day-ahead weather forecasts ("what Open-Meteo would have said").

Each hour a forecast is issued for the next ``horizon`` hours with Open-Meteo variable names:
``wind_speed_10m`` [m/s], ``shortwave_radiation`` [W/m², hourly mean], ``temperature_2m`` [C].

Error model
-----------
NWP 10 m wind is not rooftop wind: truth_10m = rooftop_hub / r (r = ``forecast_10m_to_rooftop_ratio``);
the ML must learn this *rooftop correction*. On top, for valid hour v and lead L:

    err(v, L) = σ(L) [ c ζ(v) + √(1-c²) ξ_issue(L) ],   σ(L) = σ24 (L/24)^g

ζ(v) is an AR(1) process in valid time shared by all issues (a weather situation the model gets wrong
persistently), ξ is AR(1) along lead time and fresh per issue (so successive issues *revise* the
forecast smoothly rather than jumping). A constant bias represents the NWP roof-level over-estimate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .weather import WeatherModel, _ar1

SHARED_ERROR_WEIGHT = 0.7
MIN_LEAD_H = 1


def issue_forecasts(weather: WeatherModel, n_hours: int, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forecasts issued every ``forecast_issue_every_h`` hours over the first ``n_hours`` of the run.

    Returns
    -------
    (forecasts, hourly_truth) -- forecasts have one row per (issue_time, valid_time); hourly_truth has the
    realised hourly means (hub wind, 10 m-equivalent wind, GHI, temperature).
    """
    sim = weather.site["simulation"]
    fe = sim["forecast_error"]
    horizon = int(sim["forecast_horizon_h"])
    every = int(sim["forecast_issue_every_h"])
    ratio = weather.site["wind"]["forecast_10m_to_rooftop_ratio"]
    rng = np.random.default_rng(seed + 49979687)
    truth = weather.hourly_truth(n_hours + horizon + 1)
    n_tot = truth["wind_hub"].size
    w10 = truth["wind_hub"] / ratio
    rho = fe["error_autocorr_h"]
    zeta_w = _ar1(rng.standard_normal(), rho, rng.standard_normal(n_tot))
    zeta_g = _ar1(rng.standard_normal(), rho, rng.standard_normal(n_tot))
    zeta_t = _ar1(rng.standard_normal(), rho, rng.standard_normal(n_tot))
    c = SHARED_ERROR_WEIGHT
    s = np.sqrt(1 - c * c)
    rows = []
    leads = np.arange(MIN_LEAD_H, horizon + 1)
    growth = (leads / 24.0) ** fe["wind_sigma_growth"]
    for issue in range(0, n_hours, every):
        valid = issue + leads
        ok = valid < n_tot
        v, L, gr = valid[ok], leads[ok], growth[ok]
        xi = [_ar1(rng.standard_normal(), rho, rng.standard_normal(v.size)) for _ in range(3)]
        e_w = fe["wind_sigma_ms_at_24h"] * gr * (c * zeta_w[v] + s * xi[0])
        e_g = fe["ghi_sigma_frac_at_24h"] * gr * (c * zeta_g[v] + s * xi[1])
        e_t = fe["temp_sigma_c_at_24h"] * gr * (c * zeta_t[v] + s * xi[2])
        rows.append(pd.DataFrame({
            "issue_time": truth["time_utc"][issue], "valid_time": truth["time_utc"][v], "lead_h": L,
            "wind_speed_10m": np.maximum(0.0, w10[v] + fe["wind_bias_ms"] + e_w),
            "shortwave_radiation": np.maximum(0.0, truth["ghi"][v] * (1 + e_g)),
            "temperature_2m": truth["t_amb"][v] + e_t}))
    fc = pd.concat(rows, ignore_index=True)
    for col in ("issue_time", "valid_time"):
        fc[col] = pd.to_datetime(fc[col]).dt.tz_localize("UTC")
    ht = pd.DataFrame({"valid_time": pd.to_datetime(truth["time_utc"]).tz_localize("UTC"),
                       "wind_hub": truth["wind_hub"], "wind_10m_equiv": w10, "ghi": truth["ghi"],
                       "t_amb": truth["t_amb"]})
    return fc, ht.iloc[:n_hours + horizon].reset_index(drop=True)
