"""Sensor-quirk realism: quantisation, glitch codes, debounce, stale values, network."""
import numpy as np
import pytest

from aerosolar.sim.sensors import SensorSuite, anemometer_counts


def _suite(cfg, **over):
    return SensorSuite(cfg.with_overrides(sensors=over).sensors, cfg.turbine, seed=3)


def test_anemometer_stall_quantisation_and_debounce(cfg):
    a = cfg.sensors["anemometer"]
    dt = 0.1

    def reading(v, secs=120):
        c, *_ = anemometer_counts(np.full(secs * 10, v), dt, v, 0.0, -1.0, 0.0, a["distance_constant_m"],
                                  a["overspeed_ratio"], a["stall_ms"], a["m_per_s_per_hz"], a["debounce_s"])
        return c[10:] * a["m_per_s_per_hz"]

    assert reading(0.3).max() == 0.0                                           # stalled cups
    r6 = reading(6.0)
    counts = r6 / a["m_per_s_per_hz"]
    assert np.allclose(counts, np.round(counts))                               # integer pulses per gate
    assert r6.mean() == pytest.approx(6.0, abs=0.1)
    r20 = reading(20.0)
    assert r20.max() <= 20 * a["m_per_s_per_hz"] + 1e-9                        # 50 ms lockout: <= 20 pulses/s
    assert r20.mean() < 14.0                                                   # and it under-reads strong wind


def test_ds18b20_resolution_and_glitch_codes(cfg):
    s = _suite(cfg, ds18b20={"p_power_on_85": 0.01, "p_disconnect_start": 0.002})
    n = 20000
    t = np.arange(n, dtype=float)
    out = s._ds18b20(t, {"t_gen_c": np.full(n, 40.0), "t_bearing_c": 30 + np.sin(t / 500), "t_amb_c": np.full(n, 35.0)})
    for v in out.values():
        normal = v[(v != 85.0) & (v != -127.0)]
        assert np.allclose(normal / 0.0625, np.round(normal / 0.0625))
    allv = np.concatenate(list(out.values()))
    assert (allv == 85.0).sum() > 50
    assert (allv == -127.0).sum() > 10


def test_hall_rpm_is_quantised(cfg):
    s = _suite(cfg)
    dt = 0.1
    w = 150 * 2 * np.pi / 60
    theta = w * dt * np.arange(1, 601)
    edges, _, rpm = s.hall(0.0, w, theta, np.full(600, w), 0.0, dt, 60)
    assert np.all(rpm % 60 == 0)                                               # 1 pulse/rev, 1 s gate
    assert rpm.mean() == pytest.approx(150, rel=0.05)
    assert np.allclose(np.diff(edges), 0.4, atol=1e-6)                         # exact true edges at 150 rpm


def test_network_loss_duplicates_and_clock(cfg):
    s = _suite(cfg, network={"p_duplicate": 0.02})
    n = 36000
    t = np.arange(n, dtype=float)
    start = np.datetime64("2026-06-01T00:00:00")
    df = s.packetize(t, start, {"wind_ms": np.zeros(n)})
    uniq = df["seq"].nunique()
    assert 0.0 < 1 - uniq / n < 0.05
    assert len(df) > uniq                                                      # duplicates delivered
    offs = (df["ts"].to_numpy() - start.astype("datetime64[ms]").astype(np.int64)) / 1000 - df["seq"]
    assert 0.001 < offs.abs().max() < 1.0                                      # ESP32 clock offset, bounded


def test_dht_stale_and_bh1750_saturation(cfg):
    s = _suite(cfg)
    n = 600
    t = np.arange(n, dtype=float)
    z = np.zeros(n)
    out = s.measure(t, {"v_bus": z + 13, "i_wind": z, "p_pv": z, "v_batt": z + 13, "soc": z + .5, "chg_state": z,
                        "t_gen": z + 30, "t_bearing": z + 30},
                    z, z, z, z + 800, 30 + np.linspace(0, 5, n), z + 50, z, z)
    assert (np.diff(out["dht_t_c"]) == 0).mean() >= 0.5
    assert (out["lux"] == 65535).all()                                         # 800 W/m2 saturates the BH1750
