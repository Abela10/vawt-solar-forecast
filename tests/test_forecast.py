"""Power-forecast pipeline: no look-ahead leakage, interval calibration, metric definitions."""
import numpy as np
import pandas as pd
import pytest

from aerosolar.ml.forecast_features import FEATURES, build_table
from aerosolar.ml.metrics import daily_energy_accuracy, interval_metrics, point_metrics

T0 = pd.Timestamp("2026-03-01T00:00:00", tz="UTC")


def _synthetic(n_hours=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = T0 + pd.to_timedelta(np.arange(n_hours), unit="h")
    hourly = pd.DataFrame({"wind_ms": rng.uniform(1, 8, n_hours), "p_wind": rng.uniform(0, 5, n_hours),
                           "p_pv": rng.uniform(0, 15, n_hours), "soc": rng.uniform(20, 90, n_hours),
                           "brake_frac": 0.0, "float_frac": 0.0, "ghi_clear": rng.uniform(0, 900, n_hours)}, index=idx)
    hourly["p_total"] = hourly["p_wind"] + hourly["p_pv"]
    rows = []
    for i in range(0, n_hours - 48, 1):
        it = idx[i]
        for L in range(1, 49):
            rows.append({"issue_time": it, "valid_time": it + pd.Timedelta(hours=L), "lead_h": L,
                         "wind_speed_10m": rng.uniform(1, 9), "shortwave_radiation": rng.uniform(0, 900),
                         "temperature_2m": 30.0})
    return hourly, pd.DataFrame(rows)


def test_features_do_not_see_the_future(cfg):
    hourly, fc = _synthetic()
    X1 = build_table(fc, hourly, cfg)
    issue = pd.Timestamp(fc["issue_time"].iloc[5000])
    # corrupt every measurement from the issue hour onwards: features of that issue must not change
    h2 = hourly.copy()
    h2.loc[h2.index >= issue, ["wind_ms", "p_wind", "p_pv", "soc", "p_total"]] = -999.0
    X2 = build_table(fc, h2, cfg)
    sel = X1["issue_time"] == issue
    obs = [c for c in FEATURES if c.startswith(("obs_", "pers_", "pv_yield", "wind_bias"))]
    pd.testing.assert_frame_equal(X1.loc[sel, obs].reset_index(drop=True), X2.loc[sel, obs].reset_index(drop=True))
    # ...while the targets (future actuals) do change
    assert (X2.loc[sel, "y_pv"] == -999.0).all()


def test_point_metrics_definitions():
    y = np.array([0, 0, 10, 20, 10.0])
    p = np.array([0, 1, 12, 18, 10.0])
    m = point_metrics(y, p, cap=20.0)
    assert m["mae_w"] == pytest.approx(1.0)
    assert m["cap_accuracy_mae"] == pytest.approx(0.95)
    assert m["daytime_accuracy"] == pytest.approx(1 - 4 / 40)      # hours with y >= 2 W
    assert m["r2"] == pytest.approx(1 - 9 / np.sum((y - y.mean()) ** 2))


def test_daily_energy_and_interval_metrics():
    f = pd.DataFrame({"local_date": np.repeat(["d1", "d2"], 24), "y": np.r_[np.full(24, 1.0), np.full(24, 2.0)],
                      "p": np.r_[np.full(24, 1.1), np.full(24, 2.0)]})
    d = daily_energy_accuracy(f, "y", "p")
    assert d["days"] == 2 and d["daily_energy_accuracy_mean"] == pytest.approx((0.9 + 1.0) / 2)
    i = interval_metrics(np.array([1, 2, 3, 4]), np.array([0, 0, 0, 5]), np.array([2, 2, 2, 6]))
    assert i["coverage"] == pytest.approx(0.5)
