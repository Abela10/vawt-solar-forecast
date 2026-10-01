"""Energy conservation of the physics kernel and charge conservation of the battery."""
import numpy as np
import pytest

from aerosolar.sim import kernel as K
from aerosolar.sim.electrical import COULOMBIC_EFF

from .conftest import run_constant_wind


def _turbulent_run(turbine, seconds=3600, seed=0):
    rng = np.random.default_rng(seed)
    prm = K.build_params(turbine)
    st, vt = K.ocv_tables(turbine)
    sub = int(prm[K.P_SUBSTEPS])
    x = np.zeros(seconds * sub)
    e = rng.standard_normal(x.size)
    for i in range(1, x.size):
        x[i] = 0.98 * x[i - 1] + 0.2 * e[i]
    wind = np.clip(6.5 + 1.4 * x / x.std(), 0, None)
    inp = K.default_inputs(seconds)
    inp[:, K.I_WIND_MEAS] = wind.reshape(seconds, sub).mean(axis=1)
    inp[:, K.I_PPV_AVAIL] = 10.0
    state = K.initial_state(turbine, 25.0)
    out = np.zeros((seconds, K.N_OUTPUTS))
    K.run_kernel(prm, st, vt, wind, inp, state, out, np.zeros(seconds * sub), np.zeros(seconds * sub))
    return out, prm


def test_mechanical_energy_balance(cfg):
    t = cfg.with_overrides(turbine={"brake": {"overspeed_rpm": 1e6, "gust_trip_ms": 1e6}}).turbine
    out, _ = _turbulent_run(t)
    e_in = out[:, K.O_P_AERO].sum()
    sinks = (out[:, K.O_P_FRIC] + out[:, K.O_P_BRAKE] + out[:, K.O_P_GEAR_LOSS] + out[:, K.O_P_GEN_MECH]
             + out[:, K.O_P_CU] + out[:, K.O_P_DIODE] + out[:, K.O_P_WIND_OUT]).sum()
    d_ke = out[-1, K.O_KE]   # started from rest
    assert e_in > 1000.0     # the rotor did real work [J]
    assert (sinks + d_ke) == pytest.approx(e_in, rel=0.02)


def test_battery_charge_conservation(cfg):
    out, prm = _turbulent_run(cfg.turbine, seed=1)
    i = out[:, K.O_I_BATT]
    q = np.where(i > 0, i * COULOMBIC_EFF, i).sum()               # [A s]
    cap = prm[K.P_CAP_AH] * 3600
    soc0 = cfg.turbine["electrical"]["battery"]["soc_initial"]
    assert out[-1, K.O_SOC] - soc0 == pytest.approx(q / cap, abs=1e-6)


def test_generator_power_split(cfg):
    """Converted power = delivered + diode + copper (no energy appears or vanishes in the rectifier)."""
    out, _ = run_constant_wind(cfg.turbine, 7.0, 200)
    o = out[-50:].mean(axis=0)
    p_em = o[K.O_P_WIND_OUT] + o[K.O_P_DIODE] + o[K.O_P_CU]
    assert p_em == pytest.approx(o[K.O_E_DC] * o[K.O_I_WIND], rel=0.02)
