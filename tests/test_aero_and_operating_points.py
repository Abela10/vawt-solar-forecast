"""Cp curve and steady operating points against the design table (spec section 1)."""
import numpy as np
import pytest

from aerosolar.sim import kernel as K
from aerosolar.sim.aero import AeroParams, aero_power, aero_torque, cp_of_tsr, thrust

from .conftest import run_constant_wind

RHO = 1.225


def rpm_to_w(rpm):
    return rpm * 2 * np.pi / 60


def test_cp_curve_through_design_point(cfg):
    p = AeroParams.from_config(cfg.turbine)
    assert cp_of_tsr(0.785, p) == pytest.approx(0.20)
    tsr = np.linspace(0, 1.3, 1301)
    assert tsr[np.argmax(cp_of_tsr(tsr, p))] == pytest.approx(0.785, abs=0.002)   # design point is the peak
    assert cp_of_tsr(0.0, p) == 0.0
    assert cp_of_tsr(p.runaway_tsr, p) == pytest.approx(0.0, abs=1e-9)
    assert 0.3 < p.cq_static < 0.5


@pytest.mark.parametrize("v,rpm,torque,power", [(4, 100, 0.19, 2.0), (6, 150, 0.43, 6.7), (8, 200, 0.76, 15.8)])
def test_design_table_without_cones(cfg, v, rpm, torque, power):
    t = cfg.with_overrides(turbine={"rotor": {"cones": {"enabled": False}}}).turbine
    p = AeroParams.from_config(t)
    assert aero_torque(v, rpm_to_w(rpm), RHO, p) == pytest.approx(torque, rel=0.03)
    assert aero_power(v, rpm_to_w(rpm), RHO, p) == pytest.approx(power, rel=0.03)


def test_cones_and_thrust(cfg):
    p = AeroParams.from_config(cfg.turbine)
    assert aero_torque(8, rpm_to_w(200), RHO, p) == pytest.approx(0.89, rel=0.01)
    assert aero_power(8, rpm_to_w(200), RHO, p) == pytest.approx(18.7, rel=0.01)
    assert thrust(8, RHO, p) == pytest.approx(10.9, rel=0.01)
    assert thrust(12, RHO, p) == pytest.approx(24.5, rel=0.01)


def test_steady_state_at_8ms_matches_table(cfg):
    out, _ = run_constant_wind(cfg.turbine, 8.0, 300, t_amb=25.0)
    last = out[-30:].mean(axis=0)
    assert last[K.O_RPM] == pytest.approx(200, rel=0.03)
    assert last[K.O_T_AERO] == pytest.approx(0.89, rel=0.03)
    assert last[K.O_P_AERO] == pytest.approx(18.7, rel=0.03)
    eta = last[K.O_P_WIND_OUT] / last[K.O_P_AERO]
    assert 0.55 < eta < 0.70   # ~60 % chain efficiency assumption


def test_cut_in_and_standstill(cfg):
    cut_in = cfg.turbine["rotor"]["cut_in_target_ms"]
    out, _ = run_constant_wind(cfg.turbine, cut_in, 200)
    assert out[-1, K.O_RPM] > 30, "rotor must start at the cut-in target"
    out, _ = run_constant_wind(cfg.turbine, 1.5, 200)
    assert out[-1, K.O_RPM] == 0.0


def test_electrical_cut_in_above_rotational(cfg):
    out, _ = run_constant_wind(cfg.turbine, 4.0, 300)
    assert out[-1, K.O_RPM] > 100 and out[-1, K.O_P_WIND_OUT] < 0.05   # spinning but not charging (EMF < V_batt)


def test_overspeed_trip(cfg):
    out, _ = run_constant_wind(cfg.turbine, 9.5, 120)
    assert out[-1, K.O_BRAKE] == 1
    assert out[-1, K.O_RPM] < 1.0
    assert out[:, K.O_BRAKE_REASON].max() == 1   # overspeed


def test_gust_trip_uses_anemometer(cfg):
    t = cfg.with_overrides(turbine={"brake": {"overspeed_rpm": 1e6}}).turbine
    out, _ = run_constant_wind(t, 12.5, 60)
    assert out[-1, K.O_BRAKE] == 1
    assert out[-1, K.O_BRAKE_REASON] == 2
