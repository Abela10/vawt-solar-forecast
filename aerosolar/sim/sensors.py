"""Sensor, firmware and network models: turns physical truth into what the Raspberry Pi actually receives.

Each model reproduces the quirks that matter to the forecasting pipeline (they are why the measured
actuals are cleaned before training):

* **3-cup anemometer** (ESP32 pulse counter, 1 s gate): cup inertia with a distance constant and
  *overspeeding* (cups accelerate faster than they decelerate, so gusty wind reads high), stall below
  the starting threshold, pulse quantisation of 0.667 m/s per count, and a 50 ms debounce lockout.
  Caution: the lockout caps the count at 20 pulses/s and, above that, *drops* pulses (a strict
  lockout accepts only every 2nd pulse between 20 and 40 Hz), so readings above ~13 m/s are wrong.
* **A3144 Hall, 1 pulse/rev**: edges from the rotor angle plus ~50 µs jitter; the 1 Hz rpm field is
  60 x (edges in the last second), i.e. ±60 rpm quantisation.
* **INA219**: 12-bit shunt/bus ADC LSBs, per-device gain error, current range clipping.
* **DS18B20 x3 on one 1-Wire bus**: probe thermal lag, 750 ms conversion latency, 0.0625 C steps,
  per-device offset, spurious 85.0 C power-on reads, -127 C disconnects (sometimes the whole bus).
* **BH1750**: lux with gain error, hard saturation at 65535 lx (≈ 546 W/m² at /120: saturates at noon).
* **4x LDR tracker bridge**: saturating (tanh) and slow (first-order) pointing-error signal.
* **DHT22**: 2 s minimum interval (every other packet repeats), failed reads repeat stale values.
* **Wi-Fi / MQTT**: Gilbert-Elliott burst loss, QoS-1 duplicates, late (out-of-order) packets, ESP32
  clock offset and drift with hourly NTP resync.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from ._jit import njit

TWO_PI = 2.0 * math.pi
DS18B20_FIELDS = ("t_gen_c", "t_bearing_c", "t_amb_c")
DS18B20_POWER_ON = 85.0
DS18B20_DISCONNECT = -127.0
MS_PER_S = 1000.0
DHT_DECIMALS = 1
MIN_WIND_FOR_CUP_TAU = 0.5


# ------------------------------------------------------------------------------------ JIT helpers
@njit
def anemometer_counts(wind10, dt, u_cup, phase, last_pulse, t0, dist_const, overspeed_ratio, stall, cal, debounce):
    """Cup-anemometer pulse counts per 1 s gate.

    Cup dynamics: du/dt = (v - u)/tau, tau = L/u (distance constant), tau_up = ratio * tau_down.
    Pulse frequency f = u / cal [Hz]; pulses closer than ``debounce`` [s] to the last accepted pulse are
    rejected by the firmware.

    Returns (counts per second, u_cup, phase, last_pulse) -- the last three are carried state.
    """
    n_steps = wind10.shape[0]
    sub = int(round(1.0 / dt))
    n_sec = n_steps // sub
    counts = np.zeros(n_sec)
    for i in range(n_steps):
        v = wind10[i]
        tau = dist_const / max(u_cup, MIN_WIND_FOR_CUP_TAU)
        if v > u_cup:
            tau *= overspeed_ratio
        u_cup += (v - u_cup) * (1.0 - math.exp(-dt / tau))
        f = u_cup / cal if u_cup > stall else 0.0
        ph_new = phase + f * dt
        t_step = t0 + i * dt
        k = math.floor(phase) + 1.0
        while k <= ph_new and f > 0.0:
            tp = t_step + (k - phase) / f
            if tp - last_pulse >= debounce:
                last_pulse = tp
                sec = int(tp - t0)
                if 0 <= sec < n_sec:
                    counts[sec] += 1.0
            k += 1.0
        phase = ph_new
    return counts, u_cup, phase, last_pulse


@njit
def hall_edge_times(theta0, omega0, theta_hi, omega_hi, t0, dt, offset):
    """True times [s] at which the rotor angle crosses ``offset + 2πm`` (1 magnet).

    Within each physics step the kernel integrates with constant angular acceleration, so the crossing
    is the root of θ0 + ω0 τ + ½ α τ² = target (exact, no interpolation error).
    """
    n = theta_hi.shape[0]
    out = np.empty(n * 4 + 8)
    cnt = 0
    th_a = theta0
    w_a = omega0
    for i in range(n):
        th_b = theta_hi[i]
        w_b = omega_hi[i]
        # crossings in (th_a, th_b] counted via floor() at both ends: consistent across step boundaries
        m_a = math.floor((th_a - offset) / TWO_PI)
        m_b = math.floor((th_b - offset) / TWO_PI)
        if m_b > m_a:
            alpha = (w_b - w_a) / dt
            for m in range(m_a + 1, m_b + 1):
                target = offset + TWO_PI * m
                d = target - th_a
                if abs(alpha) < 1e-9:
                    tau = d / max(w_a, 1e-9)
                else:
                    disc = w_a * w_a + 2.0 * alpha * d
                    tau = (-w_a + math.sqrt(max(disc, 0.0))) / alpha
                tau = min(max(tau, 0.0), dt)
                if cnt < out.shape[0]:
                    out[cnt] = t0 + i * dt + tau
                    cnt += 1
        th_a = th_b
        w_a = w_b
    return out[:cnt]


# ------------------------------------------------------------------------------------ suite
@dataclass
class SensorSuite:
    """All sensors + ESP32 firmware + network, with per-device calibration errors and carried state."""

    sensors: dict
    turbine: dict
    seed: int = 0
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed + 15485863)
        s, r = self.sensors, self.rng
        # per-device errors (fixed for the life of the "hardware")
        self.ina_gain = 1.0 + r.normal(0, s["ina219"]["gain_error"])
        self.ina_v_gain = 1.0 + r.normal(0, s["ina219"]["gain_error"])
        self.ina_offset = r.normal(0, s["ina219"]["offset_a"])
        self.ina_pv_gain = 1.0 + r.normal(0, s["ina219_pv"]["gain_error"])
        acc = s["ds18b20"]["accuracy_c"]
        self.ds_offset = {f: r.uniform(-acc / 2, acc / 2) for f in DS18B20_FIELDS}
        self.bh_gain = 1.0 + r.normal(0, s["bh1750"]["gain_error"])
        self.dht_t_off = r.uniform(-s["dht"]["t_accuracy_c"], s["dht"]["t_accuracy_c"])
        self.dht_rh_off = r.uniform(-s["dht"]["rh_accuracy"], s["dht"]["rh_accuracy"])
        # carried state
        self.cup_u = 0.0
        self.cup_phase = 0.0
        self.cup_last = -1.0
        self.ds_state = {f: None for f in DS18B20_FIELDS}
        self.ds_prev_true = {f: None for f in DS18B20_FIELDS}
        self.ds_burst_until = {f: -1.0 for f in DS18B20_FIELDS}
        self.ldr_state = np.zeros(2)
        self.dht_last = (float("nan"), float("nan"))
        self.soc_drift = 0.0
        self.seq = 0
        self.ge_bad = False
        self.clock_offset = s["network"]["clock_offset_s"]
        self.last_sync = 0.0
        self.last_offsets = np.zeros(0)

    # ------------------------------------------------------------------ pulse sensors
    def anemometer(self, wind10: np.ndarray, dt: float, t0: float) -> tuple[np.ndarray, np.ndarray]:
        a = self.sensors["anemometer"]
        counts, self.cup_u, self.cup_phase, self.cup_last = anemometer_counts(
            wind10, dt, self.cup_u, self.cup_phase, self.cup_last, t0, a["distance_constant_m"],
            a["overspeed_ratio"], a["stall_ms"], a["m_per_s_per_hz"], a["debounce_s"])
        return counts, counts * a["m_per_s_per_hz"]

    def hall(self, theta0: float, omega0: float, theta_hi: np.ndarray, omega_hi: np.ndarray, t0: float, dt: float,
             n_sec: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Edge times (true, with jitter) and 1 Hz counts / rpm field."""
        h = self.sensors["hall"]
        edges = hall_edge_times(theta0, omega0, theta_hi, omega_hi, t0, dt, h["magnet_angle_rad"])
        edges_j = edges + self.rng.normal(0, h["edge_jitter_s"], edges.size)
        counts = np.bincount(np.clip((edges - t0).astype(int), 0, n_sec - 1), minlength=n_sec).astype(float)
        if h.get("rpm_method", "count") == "period":
            rpm = np.zeros(n_sec)
            sec = (edges_j - t0).astype(int)
            for i in range(n_sec):
                e = edges_j[sec == i]
                if e.size >= 2:
                    rpm[i] = 60.0 / np.mean(np.diff(e)) / h["pulses_per_rev"]
                else:
                    rpm[i] = counts[i] * 60.0 / h["pulses_per_rev"]
        else:
            rpm = counts * 60.0 / h["pulses_per_rev"]
        return edges, edges_j, rpm

    # ------------------------------------------------------------------ everything else
    def measure(self, t_s: np.ndarray, truth: dict[str, np.ndarray], wind_meas: np.ndarray, rpm_meas: np.ndarray,
                hall_count: np.ndarray, ghi: np.ndarray, t_amb: np.ndarray, rh: np.ndarray, err_ew: np.ndarray,
                err_ns: np.ndarray) -> dict[str, np.ndarray]:
        """Measured 1 Hz fields (before the network)."""
        s, r, n = self.sensors, self.rng, t_s.size
        out: dict[str, np.ndarray] = {"wind_ms": wind_meas, "rpm": rpm_meas, "hall_count": hall_count}
        # INA219 on the rectified wind output
        ina = s["ina219"]
        v = truth["v_bus"] * self.ina_v_gain
        out["bus_v"] = np.round(v / ina["bus_lsb_v"]) * ina["bus_lsb_v"]
        i = np.clip(truth["i_wind"] * self.ina_gain + self.ina_offset, -ina["current_range_a"], ina["current_range_a"])
        out["current_a"] = np.round(i / ina["current_lsb_a"]) * ina["current_lsb_a"]
        p_lsb = 20.0 * ina["current_lsb_a"]
        out["power_w"] = np.round(out["bus_v"] * out["current_a"] / p_lsb) * p_lsb
        pv = s["ina219_pv"]
        out["pv_power_w"] = np.round(truth["p_pv"] * self.ina_pv_gain / pv["power_lsb_w"]) * pv["power_lsb_w"]
        bm = s["battery_monitor"]
        out["batt_v"] = np.round(truth["v_batt"] / bm["batt_v_lsb"]) * bm["batt_v_lsb"]
        # coulomb-counter SOC: random-walk drift, re-zeroed when the controller reaches float (full)
        step = r.normal(0, bm["soc_drift_pct_per_day"] / math.sqrt(86400.0), n)
        drift = self.soc_drift + np.cumsum(step)
        full = truth["chg_state"] == 2
        if full.any():
            last_full = np.maximum.accumulate(np.where(full, np.arange(n), -1))
            drift = np.where(last_full >= 0, drift - drift[np.maximum(last_full, 0)], drift)
        self.soc_drift = float(drift[-1])
        out["soc_pct"] = np.round(np.clip(truth["soc"] * 100 + drift, 0, 100), 1)
        # DS18B20 x3
        out.update(self._ds18b20(t_s, {"t_gen_c": truth["t_gen"], "t_bearing_c": truth["t_bearing"],
                                       "t_amb_c": t_amb + s["ds18b20"]["ambient_radiation_error_k_per_kwm2"] * ghi / 1000.0}))
        # BH1750
        bh = s["bh1750"]
        out["lux"] = np.clip(np.round(ghi * bh["lux_per_wm2"] * self.bh_gain), 0, bh["saturation_lux"])
        # LDR bridge (saturating, slow)
        ld = s["ldr"]
        a = math.exp(-1.0 / ld["tau_s"])
        day = ghi > 5.0
        raw = np.column_stack([np.tanh(ld["gain_per_deg"] * err_ew), np.tanh(ld["gain_per_deg"] * err_ns)])
        raw = np.where(day[:, None], raw, 0.0)
        filt = np.column_stack([lfilter([1 - a], [1, -a], raw[:, j], zi=[a * self.ldr_state[j]])[0] for j in range(2)])
        self.ldr_state = filt[-1].copy()
        filt = filt + r.normal(0, ld["noise"], filt.shape) * np.where(day, 1.0, 3.0)[:, None]
        out["ldr_ew"], out["ldr_ns"] = np.round(filt[:, 0], 3), np.round(filt[:, 1], 3)
        # DHT22: new reading every 2 s, failures repeat the stale value
        dht = s["dht"]
        period = int(round(dht["min_interval_s"]))
        tt = np.round(t_amb + self.dht_t_off + r.normal(0, 0.1, n), DHT_DECIMALS)
        hh = np.round(np.clip(rh + self.dht_rh_off + r.normal(0, 0.5, n), 0, 100), DHT_DECIMALS)
        fresh = (t_s.astype(int) % period == 0) & (r.random(n) > dht["p_read_fail"])
        idx = np.maximum.accumulate(np.where(fresh, np.arange(n), -1))
        prev_t, prev_h = self.dht_last
        out["dht_t_c"] = np.where(idx >= 0, tt[np.maximum(idx, 0)], prev_t)
        out["dht_rh"] = np.where(idx >= 0, hh[np.maximum(idx, 0)], prev_h)
        self.dht_last = (float(out["dht_t_c"][-1]), float(out["dht_rh"][-1]))
        return out

    def _ds18b20(self, t_s: np.ndarray, temps: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        d, r, n = self.sensors["ds18b20"], self.rng, t_s.size
        a = math.exp(-1.0 / d["probe_tau_s"])
        out = {}
        # bus-level disconnect bursts hit all three probes at once
        bus_bad = np.zeros(n, bool)
        starts = np.flatnonzero(r.random(n) < d["p_disconnect_start"] * d["bus_dropout_fraction"])
        for st in starts:
            bus_bad[st:st + int(r.uniform(*d["disconnect_burst_s"]))] = True
        for f, temp in temps.items():
            prev = self.ds_prev_true[f] if self.ds_prev_true[f] is not None else temp[0]
            # 750 ms conversion: the value reported at t was sampled 0.75 s earlier
            lagged = (1 - d["conversion_s"]) * temp + d["conversion_s"] * np.concatenate([[prev], temp[:-1]])
            self.ds_prev_true[f] = float(temp[-1])
            zi = self.ds_state[f] if self.ds_state[f] is not None else lagged[0]
            probe = lfilter([1 - a], [1, -a], lagged, zi=[a * zi])[0]
            self.ds_state[f] = float(probe[-1])
            val = np.round((probe + self.ds_offset[f]) / d["resolution_c"]) * d["resolution_c"]
            val = np.where(r.random(n) < d["p_power_on_85"], DS18B20_POWER_ON, val)
            bad = bus_bad.copy()
            bad |= t_s < self.ds_burst_until[f]
            for st in np.flatnonzero(r.random(n) < d["p_disconnect_start"] * (1 - d["bus_dropout_fraction"])):
                end = t_s[st] + r.uniform(*d["disconnect_burst_s"])
                bad |= (t_s >= t_s[st]) & (t_s < end)
                self.ds_burst_until[f] = max(self.ds_burst_until[f], end)
            out[f] = np.where(bad, DS18B20_DISCONNECT, val)
        return out

    # ------------------------------------------------------------------ ESP32 packet + network
    def packetize(self, t_s: np.ndarray, start_utc: np.datetime64, fields: dict[str, np.ndarray]) -> pd.DataFrame:
        """Build 1 Hz ESP32 packets and pass them through the Wi-Fi/MQTT channel.

        Returns the packets *as received by the Pi* (lost ones absent, duplicates repeated, arrival order),
        with ``rx_time`` on the Pi clock and ``ts`` on the ESP32 clock.
        """
        net, r, n = self.sensors["network"], self.rng, t_s.size
        seq = self.seq + np.arange(n)
        self.seq += n
        # ESP32 clock: offset + drift, NTP resync every ntp_resync_s
        offs = np.empty(n)
        for i, t in enumerate(t_s):
            if t - self.last_sync >= net["ntp_resync_s"]:
                self.clock_offset = r.normal(0, 0.02)
                self.last_sync = t
            offs[i] = self.clock_offset + net["clock_drift_ppm"] * 1e-6 * (t - self.last_sync)
        self.last_offsets = offs
        # Gilbert-Elliott burst loss
        lost = np.zeros(n, bool)
        u1, u2 = r.random(n), r.random(n)
        bad = self.ge_bad
        for i in range(n):
            bad = (u1[i] >= net["p_bad_to_good"]) if bad else (u1[i] < net["p_good_to_bad"])
            lost[i] = u2[i] < (net["loss_bad"] if bad else net["loss_good"])
        self.ge_bad = bad
        latency = r.uniform(*net["latency_s"], n)
        late = r.random(n) < net["p_late"]
        latency = np.where(late, latency + r.uniform(*net["late_delay_s"], n), latency)
        keep = ~lost
        dup = keep & (r.random(n) < net["p_duplicate"])
        base = start_utc.astype("datetime64[ms]")
        esp_ms = (base.astype(np.int64) + np.round((t_s + offs) * MS_PER_S)).astype(np.int64)
        rx = t_s + latency
        cols = {"ts": esp_ms, "seq": seq.astype(np.int64)}
        cols.update({k: np.asarray(v, dtype=float) for k, v in fields.items()})
        df = pd.DataFrame(cols)
        df["rx_s"] = rx
        df = df[keep]
        if dup.any():
            d = df[dup[keep]].copy()
            d["rx_s"] = d["rx_s"] + r.uniform(0.2, 2.0, len(d))
            df = pd.concat([df, d])
        df = df.sort_values("rx_s", kind="stable")
        df["rx_time"] = base + (df["rx_s"].to_numpy() * MS_PER_S).astype("timedelta64[ms]")
        return df.drop(columns=["rx_s"]).reset_index(drop=True)
