"""End-to-end engine behaviour: outputs, determinism, chunking."""
import numpy as np
import pandas as pd
import pytest

from aerosolar.sim.engine import Engine


def test_step_outputs_and_determinism(cfg):
    a = Engine(cfg, "summer_30d", seed=5).step(900)
    b = Engine(cfg, "summer_30d", seed=5).step(900)
    pd.testing.assert_frame_equal(a.truth, b.truth)
    pd.testing.assert_frame_equal(a.packets, b.packets)
    assert len(a.truth) == 900 and len(a.status) == 900
    assert list(a.packets.columns[:19]) == cfg.sensors["packet"]["fields"]     # the 19 ESP32 fields
    assert a.packets.columns[-1] == "rx_time"
    assert a.truth["ts"].dt.tz is not None


def test_live_stepping_is_deterministic_and_continuous(cfg):
    """Live mode steps in small chunks: same seed + same chunking -> identical; no jumps at chunk borders.

    (Different chunk sizes draw the random streams in a different order, so they give a different -- but
    statistically identical -- realisation; that is intended.)"""
    def run():
        e = Engine(cfg, "summer_30d", seed=6, start="2026-06-01T14:00:00")
        return pd.concat([e.step(10).truth for _ in range(30)], ignore_index=True)
    a, b = run(), run()
    pd.testing.assert_frame_equal(a, b)
    d_border = np.abs(np.diff(a["rpm"]))[9::10]          # differences across chunk borders
    d_all = np.abs(np.diff(a["rpm"]))
    assert d_border.mean() < 3 * d_all.mean() + 1.0
