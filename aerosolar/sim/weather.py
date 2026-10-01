"""Synthetic weather for the HCT rooftop (25.1 N, 55.4 E): wind, turbulence, sun, temperature, humidity, dust.

Structure (two time scales)
---------------------------
1. **Hourly layer** (generated lazily in blocks of days, so a live simulation can run forever):
   * hourly-mean wind: Gaussian AR(1) (synoptic persistence) -> Gaussian copula -> Weibull(k, λ)
     inverse CDF, times a UAE diurnal profile (calm mornings, sea breeze 11:00-18:00, weaker nights),
     times a multi-day *shamal* multiplier, times a roof-level height correction (U ∝ z^α).
   * ambient temperature and RH: monthly climatology + diurnal cycle + AR noise.
   * events: shamal (strong NW wind, often dusty), dust events (irradiance attenuation, soiling),
     cloudy spells (rare).
2. **Fast layer** (10 Hz, per chunk): the hourly mean is interpolated and turbulence is added:
   * u(t) = U(t) + σ_u [ √f_s x_slow(t) + √(1-f_s) x_fast(t) ] + gusts,  σ_u = TI · U
   * x_fast: AR(1) with integral time T_L = L/U (Taylor frozen turbulence, L ≈ 25 m on a roof)
   * x_slow: AR(1) with τ ≈ 10 min (mesoscale / sea-breeze fluctuations)
   * gusts: coherent "1 - cos" events of 4-20 s with 1.5-3 σ peak.
   AR(1)-filtered noise has a Lorentzian spectrum, close enough to Kaimal for a 1 Hz controller.

Irradiance uses solar geometry (NOAA equations) with the Haurwitz clear-sky model, then cloud and dust
attenuation. Everything is reproducible from the seed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
from scipy import stats

from ._jit import njit

SECONDS_PER_HOUR = 3600
HOURS_PER_BLOCK = 24 * 7          # hourly layer is extended one week at a time
SOLAR_CONSTANT = 1367.0           # [W/m^2]
HAURWITZ_A = 1098.0               # Haurwitz clear-sky GHI coefficients
HAURWITZ_B = 0.057
MIN_WIND_FOR_TL = 0.5             # [m/s] floor used in T_L = L/U


# ------------------------------------------------------------------------------------ solar geometry
def solar_position(ts_utc: np.ndarray, lat_deg: float, lon_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Solar zenith and azimuth [deg] (NOAA approximations, ±0.5 deg).

    Parameters
    ----------
    ts_utc : datetime64 array (UTC)
    lat_deg, lon_deg : site coordinates [deg]
    """
    t = ts_utc.astype("datetime64[s]").astype(np.int64)
    years = ts_utc.astype("datetime64[Y]")
    day_of_year = (ts_utc.astype("datetime64[D]") - years).astype(np.int64) + 1
    hour_utc = (t % 86400) / 3600.0
    gamma = 2.0 * np.pi / 365.0 * (day_of_year - 1 + (hour_utc - 12.0) / 24.0)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                       - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))       # [min]
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma) - 0.006758 * np.cos(2 * gamma)
            + 0.000907 * np.sin(2 * gamma) - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    true_solar_min = hour_utc * 60.0 + eqtime + 4.0 * lon_deg
    ha = np.radians(true_solar_min / 4.0 - 180.0)
    lat = np.radians(lat_deg)
    cos_z = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    zen = np.degrees(np.arccos(np.clip(cos_z, -1, 1)))
    az = np.degrees(np.arctan2(np.sin(ha), np.cos(ha) * np.sin(lat) - np.tan(decl) * np.cos(lat))) + 180.0
    return zen, az


def clear_sky_ghi(zenith_deg: np.ndarray) -> np.ndarray:
    """Haurwitz clear-sky global horizontal irradiance [W/m^2]."""
    cz = np.cos(np.radians(zenith_deg))
    out = np.zeros_like(cz)
    up = cz > 0.01
    out[up] = HAURWITZ_A * cz[up] * np.exp(-HAURWITZ_B / cz[up])
    return out


def split_dni_dhi(ghi: np.ndarray, zenith_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Erbs decomposition of GHI into direct-normal and diffuse-horizontal [W/m^2]."""
    cz = np.cos(np.radians(zenith_deg))
    kt = np.where(cz > 0.01, ghi / (SOLAR_CONSTANT * np.maximum(cz, 0.01)), 0.0)
    kt = np.clip(kt, 0, 1)
    df = np.where(kt <= 0.22, 1.0 - 0.09 * kt,
                  np.where(kt <= 0.80, 0.9511 - 0.1604 * kt + 4.388 * kt ** 2 - 16.638 * kt ** 3 + 12.336 * kt ** 4,
                           0.165))
    dhi = df * ghi
    dni = np.where(cz > 0.05, (ghi - dhi) / np.maximum(cz, 0.05), 0.0)
    return np.maximum(dni, 0), dhi


# ------------------------------------------------------------------------------------ AR helpers
@njit
def _ar_varying(x0: float, coeff: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """Unit-variance AR(1) with per-sample coefficient: x_k = a_k x_{k-1} + sqrt(1-a_k^2) e_k."""
    out = np.empty(coeff.shape[0])
    x = x0
    for i in range(coeff.shape[0]):
        a = coeff[i]
        x = a * x + math.sqrt(1.0 - a * a) * noise[i]
        out[i] = x
    return out


def _ar1(x0: float, a: float, noise: np.ndarray) -> np.ndarray:
    return _ar_varying(x0, np.full(noise.shape[0], a), noise)


# ------------------------------------------------------------------------------------ model
@dataclass
class WeatherChunk:
    """Weather for [t0, t0 + n) seconds."""

    wind10: np.ndarray        # hub wind at the physics rate [m/s]
    wind_mean_1s: np.ndarray  # 1 s mean of wind10 [m/s]
    hourly_mean: np.ndarray   # interpolated hourly-mean wind at 1 Hz [m/s]
    t_amb: np.ndarray         # [C]
    rh: np.ndarray            # [%]
    pressure_hpa: np.ndarray
    ghi: np.ndarray           # [W/m^2]
    dni: np.ndarray
    dhi: np.ndarray
    zenith: np.ndarray        # [deg]
    azimuth: np.ndarray       # [deg]
    dust: np.ndarray          # dust intensity 0..1
    shamal: np.ndarray        # 0/1


@dataclass
class _Event:
    start_h: float
    end_h: float
    value: float
    dusty: bool = False


@dataclass
class WeatherModel:
    """Stateful, streamable weather generator.

    Parameters
    ----------
    site : ``site.yaml`` tree
    start_local : simulation start (local time, Asia/Dubai)
    seed : RNG seed
    physics_hz : fast-layer rate [Hz]
    """

    site: dict
    start_local: datetime
    seed: int = 0
    physics_hz: int = 10
    _rng: np.random.Generator = field(init=False, repr=False)
    _hours: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        w, loc = self.site["wind"], self.site["location"]
        self.utc_offset = timedelta(hours=loc["utc_offset_h"])
        self.start_utc = np.datetime64(self.start_local - self.utc_offset, "s")
        prof = np.asarray(w["diurnal_profile"], dtype=float)
        self._diurnal = prof / prof.mean()
        m = self.site["mounting"]
        hub = m["stand_height_m"] + m["rotor_midplane_above_base_m"]
        self.hub_height_m = hub
        self.height_factor = (hub / m["reference_height_m"]) ** m["roof_shear_alpha"]
        self.ti = w["turbulence_intensity"] * (hub / m["reference_height_m"]) ** m["ti_height_exponent"]
        # hourly arrays (grown lazily)
        self._z = np.zeros(0)
        self.h_wind = np.zeros(0)       # hub hourly mean [m/s]
        self.h_shamal = np.zeros(0)
        self.h_dust = np.zeros(0)
        self.h_cloud = np.zeros(0)
        self.h_tnoise = np.zeros(0)
        self.h_rhnoise = np.zeros(0)
        self._events_until_h = 0.0
        self._shamals: list[_Event] = []
        self._dusts: list[_Event] = []
        self._clouds: list[_Event] = []
        # fast-layer state
        self._x_fast = 0.0
        self._x_slow = 0.0
        self._t_cursor = 0            # seconds since start
        self._gusts: list[tuple[float, float, float]] = []   # (start_s, duration_s, amplitude_sigma)
        self.overrides: dict = {}     # live overrides from the simulator-control UI

    # ------------------------------------------------------------------ hourly layer
    def _schedule_events(self, h0: float, h1: float) -> None:
        w, sol = self.site["wind"], self.site["solar"]
        rng = self._rng
        span_days = (h1 - h0) / 24.0
        sh = w["shamal"]
        for _ in range(rng.poisson(sh["rate_per_30d"] * span_days / 30.0)):
            st = rng.uniform(h0, h1)
            dur = rng.uniform(*sh["duration_days"]) * 24.0
            self._shamals.append(_Event(st, st + dur, rng.uniform(*sh["wind_multiplier"]),
                                        dusty=rng.random() < sh["dust_probability"]))
        for _ in range(rng.poisson(sol["dust_event_rate_per_30d"] * span_days / 30.0)):
            st = rng.uniform(h0, h1)
            self._dusts.append(_Event(st, st + rng.uniform(*sol["dust_duration_h"]), rng.uniform(*sol["dust_attenuation"])))
        for day in range(int(h0 // 24), int(math.ceil(h1 / 24))):
            if rng.random() < sol["cloud_probability_per_day"]:
                st = day * 24 + rng.uniform(6, 14)
                self._clouds.append(_Event(st, st + rng.uniform(2, 8), rng.uniform(*sol["cloud_depth"])))

    def _ensure_hours(self, n_hours: int) -> None:
        """Extend the hourly layer to cover at least ``n_hours`` hours from the start."""
        while self._hours < n_hours:
            h0, h1 = self._hours, self._hours + HOURS_PER_BLOCK
            self._schedule_events(h0, h1)
            w, cl = self.site["wind"], self.site["climate"]
            rho = w["hourly_autocorr"]
            z0 = self._z[-1] if self._z.size else self._rng.standard_normal()
            z = _ar1(z0, rho, self._rng.standard_normal(HOURS_PER_BLOCK))
            u = stats.norm.cdf(z)
            base = stats.weibull_min.ppf(np.clip(u, 1e-9, 1 - 1e-9), w["weibull_k"], scale=w["weibull_scale_ms"])
            hours = np.arange(h0, h1) + 0.5
            local_hour = ((self.start_local.hour + self.start_local.minute / 60.0 + hours) % 24).astype(int)
            shamal = np.zeros(HOURS_PER_BLOCK)
            dust = np.zeros(HOURS_PER_BLOCK)
            cloud = np.zeros(HOURS_PER_BLOCK)
            for ev in self._shamals:
                m = (hours >= ev.start_h) & (hours < ev.end_h)
                # shamal: strong, persistent, weak diurnal cycle -> multiply and flatten the profile
                shamal[m] = ev.value
                if ev.dusty:
                    dust[m] = np.maximum(dust[m], 0.5)
            for ev in self._dusts:
                m = (hours >= ev.start_h) & (hours < ev.end_h)
                dust[m] = np.maximum(dust[m], ev.value / 0.45)
            for ev in self._clouds:
                m = (hours >= ev.start_h) & (hours < ev.end_h)
                cloud[m] = np.maximum(cloud[m], ev.value)
            diurnal = self._diurnal[local_hour]
            diurnal = np.where(shamal > 0, 0.5 + 0.5 * diurnal, diurnal)
            mult = np.where(shamal > 0, shamal, 1.0)
            hub = base * diurnal * mult * self.height_factor
            # soft saturation: products of a Weibull tail, the afternoon peak and a shamal multiplier would
            # otherwise produce unphysical hourly means; U_max * tanh(U / U_max) leaves typical winds untouched
            u_max = w.get("max_hourly_mean_ms", np.inf)
            if np.isfinite(u_max):
                hub = u_max * np.tanh(hub / u_max)
            self._z = np.concatenate([self._z, z])
            self.h_wind = np.concatenate([self.h_wind, hub])
            self.h_shamal = np.concatenate([self.h_shamal, (shamal > 0).astype(float)])
            self.h_dust = np.concatenate([self.h_dust, np.clip(dust, 0, 1)])
            self.h_cloud = np.concatenate([self.h_cloud, cloud])
            tn = _ar1(0.0 if not self.h_tnoise.size else self.h_tnoise[-1] / max(cl["temp_noise_c"], 1e-9), 0.9,
                      self._rng.standard_normal(HOURS_PER_BLOCK)) * cl["temp_noise_c"]
            rn = _ar1(0.0, 0.9, self._rng.standard_normal(HOURS_PER_BLOCK)) * 6.0
            self.h_tnoise = np.concatenate([self.h_tnoise, tn])
            self.h_rhnoise = np.concatenate([self.h_rhnoise, rn])
            self._hours = h1

    def hourly_times_utc(self, n_hours: int) -> np.ndarray:
        return self.start_utc + np.arange(n_hours).astype("timedelta64[h]")

    # ------------------------------------------------------------------ climatology helpers
    def _climate(self, t_rel_s: np.ndarray, ts_utc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Ambient temperature [C] and RH [%] at times ``t_rel_s`` (seconds since start)."""
        cl = self.site["climate"]
        local = ts_utc + np.timedelta64(int(self.utc_offset.total_seconds()), "s")
        month_float = _month_float(local)
        tm = _interp_monthly(np.asarray(cl["monthly_mean_t_c"], float), month_float)
        amp = _interp_monthly(np.asarray(cl["diurnal_amplitude_c"], float), month_float)
        rhm = _interp_monthly(np.asarray(cl["monthly_mean_rh"], float), month_float)
        hod = (local.astype("datetime64[s]").astype(np.int64) % 86400) / 3600.0
        diurnal = np.cos(2 * np.pi * (hod - cl["temp_peak_hour"]) / 24.0)
        h = t_rel_s / SECONDS_PER_HOUR
        tnoise = np.interp(h, np.arange(self.h_tnoise.size) + 0.5, self.h_tnoise)
        rnoise = np.interp(h, np.arange(self.h_rhnoise.size) + 0.5, self.h_rhnoise)
        shamal = np.interp(h, np.arange(self.h_shamal.size) + 0.5, self.h_shamal)
        t_amb = tm + amp * diurnal + tnoise - 1.5 * shamal
        rh = rhm - cl["rh_diurnal_amplitude"] * diurnal + rnoise - 15.0 * shamal
        return t_amb, np.clip(rh, 5.0, 98.0)

    # ------------------------------------------------------------------ fast layer
    def generate(self, n_seconds: int) -> WeatherChunk:
        """Generate the next ``n_seconds`` of weather (continues from the previous call)."""
        w = self.site["wind"]
        hz = self.physics_hz
        t0 = self._t_cursor
        self._ensure_hours(int(math.ceil((t0 + n_seconds) / SECONDS_PER_HOUR)) + 49)
        t_fast = t0 + (np.arange(n_seconds * hz) + 0.5) / hz
        t_1s = t0 + np.arange(n_seconds) + 0.5
        hour_centres = np.arange(self.h_wind.size) + 0.5
        U = np.interp(t_fast / SECONDS_PER_HOUR, hour_centres, self.h_wind)
        U = U * self.overrides.get("wind_multiplier", 1.0)
        if "wind_mean_ms" in self.overrides:
            U = np.full_like(U, self.overrides["wind_mean_ms"])
        ti = self.overrides.get("turbulence_intensity", self.ti)
        sigma = ti * np.maximum(U, 0.3)
        # fast turbulence: a = exp(-dt U / L)
        a_fast = np.exp(-(1.0 / hz) * np.maximum(U, MIN_WIND_FOR_TL) / w["integral_length_m"])
        x_fast = _ar_varying(self._x_fast, a_fast, self._rng.standard_normal(U.size))
        self._x_fast = float(x_fast[-1])
        a_slow = math.exp(-1.0 / w["slow_time_constant_s"])
        x_slow_1s = _ar1(self._x_slow, a_slow, self._rng.standard_normal(n_seconds))
        self._x_slow = float(x_slow_1s[-1])
        x_slow = np.repeat(x_slow_1s, hz)
        fs = w["slow_fraction"]
        u = U + sigma * (math.sqrt(fs) * x_slow + math.sqrt(1 - fs) * x_fast)
        u += self._gust_signal(t_fast, sigma, n_seconds, t0)
        u = np.maximum(u, 0.0)

        ts_utc = self.start_utc + np.round(t_1s - 0.5).astype("timedelta64[s]")
        t_amb, rh = self._climate(t_1s, ts_utc)
        if "t_amb_offset_c" in self.overrides:
            t_amb = t_amb + self.overrides["t_amb_offset_c"]
        zen, az = solar_position(ts_utc, self.site["location"]["latitude_deg"], self.site["location"]["longitude_deg"])
        h = t_1s / SECONDS_PER_HOUR
        dust = np.interp(h, hour_centres, self.h_dust)
        cloud = np.interp(h, hour_centres, self.h_cloud)
        dust = np.maximum(dust, self.overrides.get("dust", 0.0))
        shamal = np.interp(h, hour_centres, self.h_shamal)
        sol = self.site["solar"]
        att_dust = dust * sol["dust_attenuation"][1]
        ghi = clear_sky_ghi(zen) * sol["linke_like_clear_factor"] * (1 - cloud) * (1 - att_dust)
        dni, dhi = split_dni_dhi(ghi, zen)
        self._t_cursor = t0 + n_seconds
        return WeatherChunk(wind10=u, wind_mean_1s=u.reshape(n_seconds, hz).mean(axis=1),
                            hourly_mean=U.reshape(n_seconds, hz).mean(axis=1), t_amb=t_amb, rh=rh,
                            pressure_hpa=np.full(n_seconds, self.site["climate"]["pressure_hpa"]),
                            ghi=ghi, dni=dni, dhi=dhi, zenith=zen, azimuth=az, dust=dust,
                            shamal=(shamal > 0.5).astype(float))

    def _gust_signal(self, t_fast: np.ndarray, sigma: np.ndarray, n_seconds: int, t0: int) -> np.ndarray:
        """Coherent 1-cos gusts. Events are scheduled per chunk and may overlap the next chunk."""
        w = self.site["wind"]
        rate = w["gust_rate_per_h"] * self.overrides.get("gust_rate_multiplier", 1.0)
        for _ in range(self._rng.poisson(rate * n_seconds / SECONDS_PER_HOUR)):
            self._gusts.append((t0 + self._rng.uniform(0, n_seconds), self._rng.uniform(*w["gust_duration_s"]),
                                self._rng.uniform(*w["gust_amplitude_sigma"])))
        g = np.zeros_like(t_fast)
        keep = []
        t_end = t0 + n_seconds
        for st, dur, amp in self._gusts:
            m = (t_fast >= st) & (t_fast < st + dur)
            if m.any():
                ph = (t_fast[m] - st) / dur
                g[m] += amp * sigma[m] * 0.5 * (1 - np.cos(2 * np.pi * ph))
            if st + dur > t_end:
                keep.append((st, dur, amp))
        self._gusts = keep
        return g

    # ------------------------------------------------------------------ truth for forecasts
    def hourly_truth(self, n_hours: int) -> dict[str, np.ndarray]:
        """Hourly means (hub wind, GHI, temperature) used as 'what actually happened' for forecasts."""
        self._ensure_hours(n_hours)
        t_rel = (np.arange(n_hours) + 0.5) * SECONDS_PER_HOUR
        ts = self.start_utc + np.arange(n_hours).astype("timedelta64[h]") + np.timedelta64(30, "m")
        t_amb, rh = self._climate(t_rel, ts)
        # hourly mean GHI from a 5-minute grid
        sub = 12
        ts5 = (self.start_utc + (np.arange(n_hours * sub) * 300).astype("timedelta64[s]")
               + np.timedelta64(150, "s"))
        zen, _ = solar_position(ts5, self.site["location"]["latitude_deg"], self.site["location"]["longitude_deg"])
        att = (1 - np.repeat(self.h_cloud[:n_hours], sub)) * (1 - np.repeat(self.h_dust[:n_hours], sub)
                                                               * self.site["solar"]["dust_attenuation"][1])
        ghi = (clear_sky_ghi(zen) * att).reshape(n_hours, sub).mean(axis=1)
        return {"time_utc": self.hourly_times_utc(n_hours), "wind_hub": self.h_wind[:n_hours].copy(),
                "ghi": ghi, "t_amb": t_amb, "rh": rh}


def _month_float(local: np.ndarray) -> np.ndarray:
    """Month as a continuous number (0 = mid-January) for smooth climatology interpolation."""
    d = local.astype("datetime64[D]")
    m = d.astype("datetime64[M]")
    day = (d - m).astype(np.int64)
    days_in_month = ((m + 1).astype("datetime64[D]") - m.astype("datetime64[D]")).astype(np.int64)
    return (m.astype(np.int64) % 12) + (day + 0.5) / days_in_month - 0.5


def _interp_monthly(values12: np.ndarray, month_float: np.ndarray) -> np.ndarray:
    ext = np.concatenate([values12[-1:], values12, values12[:1]])
    return np.interp(month_float, np.arange(-1, 13), ext)


def weibull_fit(samples: np.ndarray) -> tuple[float, float]:
    """Maximum-likelihood Weibull (k, λ) for positive samples."""
    x = samples[samples > 0.05]
    k, _, lam = stats.weibull_min.fit(x, floc=0)
    return float(k), float(lam)
