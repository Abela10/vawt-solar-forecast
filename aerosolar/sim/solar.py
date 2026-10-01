"""Tracking PV panel, sun tracker and soiling.

Plane-of-array irradiance for a 2-axis tracker with pointing error δ [deg]:

    POA = DNI cos δ + DHI (1 + cos β)/2          (β = panel tilt = sun zenith when tracking)

Cell temperature (Faiman):  T_cell = T_amb + POA / (u0 + u1 v)                       [C]
DC power after MPPT:

    P = P_rated (POA / 1000) (1 + γ (T_cell - 25)) · derate · (1 - soiling)                       [W]

At STC this is 18 W x 0.90 = 16.2 W, the design-table value. In UAE summer T_cell ≈ 65-70 C costs
another ~17 %.

Soiling accumulates daily (0.2-0.5 %/day typical in the UAE), much faster during dust events, and is
reset by periodic cleaning. The rotor blades/cones soil too (slow Cp loss).
Tracker: the 4-LDR tracker follows the sun with ~1 deg RMS pointing error and parks facing east at night.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import lfilter

FAIMAN_U0 = 25.0      # [W/(m^2 K)]
FAIMAN_U1 = 6.84      # [W s/(m^3 K)]
STC_IRRADIANCE = 1000.0
STC_TEMP_C = 25.0
SECONDS_PER_DAY = 86400.0
NIGHT_ZENITH_DEG = 90.0
PARK_AZIMUTH_DEG = 90.0   # tracker parks facing east at night
PARK_ZENITH_DEG = 60.0
TRACKER_ERR_TAU_S = 120.0
CLEANING_LOCAL_HOUR = 8


@dataclass
class PVOutput:
    p_avail: np.ndarray       # MPPT output available [W]
    poa: np.ndarray           # plane-of-array irradiance [W/m^2]
    t_cell: np.ndarray        # [C]
    err_ew_deg: np.ndarray    # tracker pointing error, east-west component [deg]
    err_ns_deg: np.ndarray    # north-south component [deg]
    soiling: np.ndarray       # natural soiling loss fraction
    rotor_soiling: np.ndarray  # natural rotor Cp loss fraction


@dataclass
class PVModel:
    turbine: dict
    site: dict
    seed: int = 0
    soiling: float = 0.0
    rotor_soiling: float = 0.0
    _rng: np.random.Generator = field(init=False, repr=False)
    _err_state: np.ndarray = field(init=False, repr=False)
    _last_clean_day: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed + 7919)
        self._err_state = np.zeros(2)

    def step(self, t_rel_s: np.ndarray, local_hour: np.ndarray, zen: np.ndarray, az: np.ndarray,
             dni: np.ndarray, dhi: np.ndarray, t_amb: np.ndarray, wind: np.ndarray, dust: np.ndarray) -> PVOutput:
        """Advance the PV subsystem over a chunk of 1 Hz samples.

        Parameters
        ----------
        t_rel_s : seconds since the simulation start
        local_hour : local clock hour [0, 24)
        zen, az : sun zenith / azimuth [deg]; dni, dhi : [W/m^2]; t_amb : [C]; wind : [m/s]
        dust : dust intensity 0..1 (raises the soiling rate)
        """
        pv, sol = self.turbine["pv"], self.site["solar"]["soiling"]
        n = t_rel_s.size
        # --- soiling (per-second integration of a per-day rate)
        rate = sol["base_rate_per_day"] * (1.0 + (sol["dust_rate_multiplier"] - 1.0) * dust) / SECONDS_PER_DAY
        soil = self.soiling + np.cumsum(rate)
        rrate = sol["rotor_soiling_rate_per_day"] * (1.0 + (sol["dust_rate_multiplier"] - 1.0) * dust) / SECONDS_PER_DAY
        rsoil = self.rotor_soiling + np.cumsum(rrate)
        interval = sol["cleaning_interval_days"]
        if interval and interval > 0:
            day = (t_rel_s // SECONDS_PER_DAY).astype(int)
            clean = (day - self._last_clean_day >= interval) & (local_hour.astype(int) == CLEANING_LOCAL_HOUR)
            if clean.any():
                i = int(np.argmax(clean))
                soil[i:] -= soil[i]
                self._last_clean_day = int(day[i])
        soil = np.clip(soil, 0.0, 0.6)
        self.soiling = float(soil[-1])
        self.rotor_soiling = float(np.clip(rsoil[-1], 0.0, 0.5))

        # --- tracker pointing error (AR(1) jitter when healthy)
        a = np.exp(-1.0 / TRACKER_ERR_TAU_S)
        noise = self._rng.standard_normal((n, 2)) * pv["tracker"]["nominal_error_deg"] * np.sqrt(1 - a * a)
        err = np.column_stack([lfilter([1.0], [1.0, -a], noise[:, j], zi=[a * self._err_state[j]])[0]
                               for j in range(2)])
        self._err_state = err[-1].copy()
        day_mask = zen < NIGHT_ZENITH_DEG
        err_ew, err_ns = err[:, 0].copy(), err[:, 1].copy()
        delta = np.sqrt(err_ew ** 2 + err_ns ** 2)
        tilt = np.where(day_mask, zen, PARK_ZENITH_DEG)
        poa = dni * np.clip(np.cos(np.radians(delta)), 0, 1) + dhi * (1 + np.cos(np.radians(tilt))) / 2
        poa = np.where(day_mask, poa, 0.0)
        t_cell = t_amb + poa / (FAIMAN_U0 + FAIMAN_U1 * wind)
        p = (pv["rated_w"] * poa / STC_IRRADIANCE * (1 + pv["temp_coeff_per_k"] * (t_cell - STC_TEMP_C))
             * pv["derate"] * (1 - soil))
        return PVOutput(p_avail=np.maximum(p, 0.0), poa=poa, t_cell=t_cell, err_ew_deg=err_ew, err_ns_deg=err_ns,
                        soiling=soil, rotor_soiling=np.clip(rsoil, 0, 0.5))
