"""Dataset assembly: received packets -> clean 1 Hz grid -> 1-minute table.

Only what the Pi actually has is used as model input: received packets and Pi-side status.
``truth_1hz`` is used *only* for evaluating the simulator, never as a feature or a target.

Time alignment
--------------
Packets carry the ESP32 send time ``ts`` (offset, drifting clock) and the Pi receive time. The ESP32
offset is estimated per packet (:func:`estimate_esp32_offset`) and removed, then each packet is snapped
to the nearest second. Duplicates (QoS 1) are dropped by ``seq``; lost seconds stay NaN.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..sim.weather import clear_sky_ghi, solar_position

FIELDS_NUMERIC = ("wind_ms", "rpm", "hall_count", "bus_v", "current_a", "power_w", "pv_power_w", "batt_v",
                  "soc_pct", "t_gen_c", "t_bearing_c", "t_amb_c", "lux", "ldr_ew", "ldr_ns", "dht_t_c", "dht_rh")
MINUTE = 60


def estimate_esp32_offset(ts_esp_s: np.ndarray, rx_pi_s: np.ndarray, window: int = 61, quantile: float = 0.9,
                          latency_floor_s: float = 0.02) -> np.ndarray:
    """ESP32-minus-Pi clock offset [s] per packet, from packet send stamps vs Pi receive stamps.

    d = ts_esp - rx_pi = offset - latency. Latency is always positive and only rarely large (late
    packets), so a high rolling quantile of d tracks ``offset - small latency``; adding the typical
    minimum latency removes most of the bias. A centred window follows the hourly NTP re-syncs.
    """
    order = np.argsort(ts_esp_s)
    d = pd.Series(np.asarray(ts_esp_s, float)[order] - np.asarray(rx_pi_s, float)[order])
    q = d.rolling(window, center=True, min_periods=max(5, window // 4)).quantile(quantile)
    off = q.bfill().ffill().to_numpy() + latency_floor_s
    out = np.empty_like(off)
    out[order] = off
    return out


def run_meta(run: Path) -> dict:
    return json.loads((Path(run) / "meta.json").read_text(encoding="utf-8"))


def start_utc(run: Path) -> pd.Timestamp:
    return pd.Timestamp(run_meta(run)["start_utc"]).tz_localize("UTC")


def packets_to_grid(packets: pd.DataFrame, t0_utc: pd.Timestamp, n_seconds: int | None = None) -> pd.DataFrame:
    """Received packets -> one row per second (index ``t`` = seconds since ``t0_utc``), NaN where lost.

    Uses only Pi-observable information (``ts`` ESP32 clock, ``rx_time`` Pi clock, ``seq``).
    """
    p = packets.drop_duplicates("seq", keep="first").copy()
    base_ms = t0_utc.value // 1_000_000
    ts = (p["ts"].to_numpy(np.int64) - base_ms) / 1000.0
    rx_ms = pd.to_datetime(p["rx_time"]).to_numpy().astype("datetime64[ms]").astype(np.int64)
    rx = (rx_ms - base_ms) / 1000.0
    off = estimate_esp32_offset(ts, rx)
    t = np.round(ts - off).astype(np.int64)
    p["t"] = t
    p = p.sort_values("seq").drop_duplicates("t", keep="last")
    n = int(n_seconds if n_seconds is not None else t.max() + 1)
    grid = p.set_index("t").reindex(np.arange(n))
    grid.index.name = "t"
    return grid[[c for c in FIELDS_NUMERIC if c in grid.columns]].astype(float)


def load_run_1hz(run: Path) -> pd.DataFrame:
    """Pi-observable 1 Hz frame (packets on the grid + controller status), index t [s]."""
    run = Path(run)
    status = pd.read_parquet(run / "status_1hz.parquet")
    n = len(status)
    grid = packets_to_grid(pd.read_parquet(run / "telemetry_1hz.parquet"), start_utc(run), n)
    st = status.set_index(status["t"].astype(np.int64))[["brake", "chg_state", "bms_cut", "load_shed"]]
    df = grid.join(st)
    df["ts"] = start_utc(run) + pd.to_timedelta(df.index, unit="s")
    return df


def solar_context(ts_utc: pd.Series | pd.DatetimeIndex, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Clear-sky GHI [W/m^2] and sun zenith [deg] -- deterministic, available on the Pi from the clock."""
    loc = cfg.site["location"]
    t = pd.DatetimeIndex(ts_utc).tz_convert("UTC").tz_localize(None).to_numpy().astype("datetime64[s]")
    zen, _ = solar_position(t, loc["latitude_deg"], loc["longitude_deg"])
    return clear_sky_ghi(zen), zen


def minute_table(df1: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """1-minute aggregates of the cleaned 1 Hz stream + time/solar context (index = minute number).

    Irradiance from the BH1750 (lux/120) saturates near 546 W/m²; the clear-sky GHI from solar geometry plus the
    saturation fraction lets the models work around it.
    """
    lux_per = cfg.sensors["bh1750"]["lux_per_wm2"]
    sat = cfg.sensors["bh1750"]["saturation_lux"]
    d = df1.copy()
    d["minute"] = d.index // MINUTE
    d["lux_sat"] = (d["lux"] >= sat).astype(float).where(d["lux"].notna())
    agg = {c: "mean" for c in FIELDS_NUMERIC if c in d.columns}
    g = d.groupby("minute")
    m = g.agg(agg)
    m["wind_std"] = g["wind_ms"].std()
    m["n_valid"] = g["power_w"].count()
    m["brake_frac"] = g["brake"].mean()
    m["chg_state"] = g["chg_state"].max()          # float/dump anywhere in the minute matters most
    m["lux_sat_frac"] = g["lux_sat"].mean()
    m["ts"] = g["ts"].first()
    m["irr"] = m["lux"] / lux_per
    ghi_cs, zen = solar_context(m["ts"], cfg)
    m["ghi_clear"] = ghi_cs
    m["sun_zenith"] = zen
    local = m["ts"].dt.tz_convert(None) + pd.Timedelta(hours=cfg.site["location"]["utc_offset_h"])
    hod = local.dt.hour + local.dt.minute / 60.0
    m["hod_sin"] = np.sin(2 * np.pi * hod / 24)
    m["hod_cos"] = np.cos(2 * np.pi * hod / 24)
    m["t_ref"] = m["dht_t_c"].where(m["dht_t_c"].notna(), m["t_amb_c"])
    return m
