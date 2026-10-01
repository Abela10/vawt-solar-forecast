"""Simulation engine: weather → PV → physics kernel → sensors → network.

The engine is *streaming*: :meth:`Engine.step` advances by any number of seconds and returns everything
produced in that interval. Batch dataset generation calls it with 1 h chunks; a live source can call it
once per second. Live controls (weather overrides, brake command, extra load) are applied between steps.

Clocks
------
* ``t`` -- true simulation time [s since start]; UTC = start_utc + t.
* Pi clock -- NTP-disciplined; treated as true time.
* ESP32 clock -- offset + drift, resynchronised hourly (see :mod:`aerosolar.sim.sensors`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from ..config import Config
from . import kernel as K
from .aero import air_density
from .electrical import CHG_NAMES
from .sensors import SensorSuite
from .solar import PVModel
from .weather import WeatherModel

SECONDS_PER_HOUR = 3600
BRAKE_REASONS = {0: "none", 1: "overspeed", 2: "gust", 3: "manual", 4: "power_loss"}


@dataclass
class StepResult:
    """Everything the twin produced in one :meth:`Engine.step`."""

    truth: pd.DataFrame          # 1 Hz physical truth (for evaluation, never a model input)
    packets: pd.DataFrame        # ESP32 packets as received by the Pi (arrival order)
    status: pd.DataFrame         # Pi-side controller status (brake, charge state), 1 Hz


class Engine:
    """The digital twin of the turbine and PV system.

    Parameters
    ----------
    cfg : loaded :class:`Config`
    preset : scenario name from ``scenarios.yaml`` (or a preset dict, or None for the site default)
    seed : master RNG seed (weather, PV and sensors derive their own streams from it)
    start : optional local start time overriding the preset (ISO string)
    """

    def __init__(self, cfg: Config, preset: str | dict | None = None, seed: int = 0, start: str | None = None):
        p = cfg.preset(preset) if isinstance(preset, str) else dict(preset or {})
        self.preset_name = preset if isinstance(preset, str) else p.get("name", "custom")
        self.cfg = cfg.with_overrides(turbine=p.get("turbine_overrides"), site=p.get("site_overrides"),
                                      sensors=p.get("sensor_overrides"))
        self.seed = seed
        c = self.cfg
        self.start_local = datetime.fromisoformat(start or p.get("start") or c.site["simulation"]["start"])
        self.duration_h = p.get("duration_h")
        self.hz = int(c.turbine["control"]["physics_hz"])
        self.dt = 1.0 / self.hz

        self.weather = WeatherModel(c.site, self.start_local, seed=seed, physics_hz=self.hz)
        self.start_utc = self.weather.start_utc
        self.pv = PVModel(c.turbine, c.site, seed=seed)
        self.sensors = SensorSuite(c.sensors, c.turbine, seed=seed)
        self.prm = K.build_params(c.turbine)
        self.soc_tab, self.v_tab = K.ocv_tables(c.turbine)
        self.weather._ensure_hours(49)
        t_amb0 = float(self.weather._climate(np.array([0.5]), np.array([self.start_utc]))[0][0])
        self.state = K.initial_state(c.turbine, t_amb0)
        self.t = 0
        # live controls
        self.brake_cmd = 0.0
        self.load_extra_w = 0.0

    # ------------------------------------------------------------------ helpers
    def utc(self, t_s: np.ndarray) -> pd.DatetimeIndex:
        base = pd.Timestamp(self.start_utc).tz_localize("UTC")
        return base + pd.to_timedelta(np.asarray(t_s, dtype=float), unit="s")

    def local_hour(self, t_s: np.ndarray) -> np.ndarray:
        h0 = self.start_local.hour + self.start_local.minute / 60.0 + self.start_local.second / 3600.0
        return (h0 + t_s / SECONDS_PER_HOUR) % 24.0

    # ------------------------------------------------------------------ main step
    def step(self, n: int) -> StepResult:
        """Advance the twin by ``n`` seconds."""
        n = int(n)
        t0 = self.t
        t_s = t0 + np.arange(n, dtype=float)
        wx = self.weather.generate(n)
        pv = self.pv.step(t_s, self.local_hour(t_s), wx.zenith, wx.azimuth, wx.dni, wx.dhi, wx.t_amb,
                          wx.wind_mean_1s, wx.dust)
        counts, wind_meas = self.sensors.anemometer(wx.wind10, self.dt, float(t0))
        rho = air_density(wx.t_amb, wx.rh, float(self.cfg.site["climate"]["pressure_hpa"]))

        inp = K.default_inputs(n)
        inp[:, K.I_RHO] = rho
        inp[:, K.I_TAMB] = wx.t_amb
        inp[:, K.I_GHI] = wx.ghi
        inp[:, K.I_WIND_MEAS] = wind_meas
        inp[:, K.I_PPV_AVAIL] = pv.p_avail
        inp[:, K.I_CP_FACTOR] = 1.0 - pv.rotor_soiling
        inp[:, K.I_BRAKE_CMD] = self.brake_cmd
        inp[:, K.I_LOAD_EXTRA_W] = self.load_extra_w

        theta0, omega0 = float(self.state[K.S_THETA]), float(self.state[K.S_OMEGA])
        out = np.zeros((n, K.N_OUTPUTS))
        theta_hi = np.zeros(n * self.hz)
        omega_hi = np.zeros(n * self.hz)
        K.run_kernel(self.prm, self.soc_tab, self.v_tab, wx.wind10, inp, self.state, out, theta_hi, omega_hi)

        ts = self.utc(t_s)
        truth = pd.DataFrame({name: out[:, i] for i, name in enumerate(K.OUTPUT_NAMES)})
        truth.insert(0, "t", t_s)
        truth.insert(0, "ts", ts)
        truth["seq"] = self.sensors.seq + np.arange(n)
        extra = {"wind_hourly_mean": wx.hourly_mean, "t_amb": wx.t_amb, "rh": wx.rh, "rho": rho, "ghi": wx.ghi,
                 "dni": wx.dni, "dhi": wx.dhi, "zenith": wx.zenith, "poa": pv.poa, "t_cell": pv.t_cell,
                 "dust": wx.dust, "shamal": wx.shamal, "pv_soiling": pv.soiling, "rotor_soiling": pv.rotor_soiling,
                 "tracker_err_deg": np.hypot(pv.err_ew_deg, pv.err_ns_deg), "p_pv_avail": pv.p_avail,
                 "wind_std_1s": wx.wind10.reshape(n, self.hz).std(axis=1), "anemometer_counts": counts}
        for k, v in extra.items():
            truth[k] = v

        # --- sensors and network
        edges, _, rpm_meas = self.sensors.hall(theta0, omega0, theta_hi, omega_hi, float(t0), self.dt, n)
        hall_count = np.bincount(np.clip((edges - t0).astype(int), 0, n - 1), minlength=n).astype(float)
        fields = self.sensors.measure(t_s, {"v_bus": out[:, K.O_V_BUS], "i_wind": out[:, K.O_I_WIND],
                                            "p_pv": out[:, K.O_P_PV], "v_batt": out[:, K.O_V_BATT],
                                            "soc": out[:, K.O_SOC], "chg_state": out[:, K.O_CHG],
                                            "t_gen": out[:, K.O_TG], "t_bearing": out[:, K.O_TB]},
                                      wind_meas, rpm_meas, hall_count, wx.ghi, wx.t_amb, wx.rh,
                                      pv.err_ew_deg, pv.err_ns_deg)
        packets = self.sensors.packetize(t_s, self.start_utc, fields)
        status = pd.DataFrame({"ts": ts, "t": t_s, "brake": out[:, K.O_BRAKE].astype(np.int8),
                               "brake_reason": out[:, K.O_BRAKE_REASON].astype(np.int8),
                               "chg_state": out[:, K.O_CHG].astype(np.int8),
                               "bms_cut": out[:, K.O_BMS_CUT].astype(np.int8),
                               "load_shed": (out[:, K.O_SOC] < self.cfg.turbine["electrical"]["self_consumption"]["shed_soc"]).astype(np.int8)})
        self.t += n
        return StepResult(truth=truth, packets=packets, status=status)


def describe_state(engine: Engine) -> dict:
    """Small human-readable snapshot (used by the CLI progress line and the API)."""
    s = engine.state
    return {"t_s": engine.t, "rpm": float(s[K.S_OMEGA] * 60 / (2 * math.pi)), "soc": float(s[K.S_SOC]),
            "brake": bool(s[K.S_BRAKE] > 0.5), "chg_state": CHG_NAMES.get(int(s[K.S_CHG]), "?")}
