import numpy as np
import pytest

from aerosolar.config import Config
from aerosolar.sim import kernel as K


@pytest.fixture(scope="session")
def cfg() -> Config:
    return Config.load()


def run_constant_wind(turbine: dict, v: float, seconds: int = 300, t_amb: float = 25.0, omega0: float = 0.0,
                      inputs: dict | None = None):
    """Run the physics kernel at constant wind; returns (out, state)."""
    prm = K.build_params(turbine)
    st, vt = K.ocv_tables(turbine)
    sub = int(prm[K.P_SUBSTEPS])
    inp = K.default_inputs(seconds)
    inp[:, K.I_WIND_MEAS] = v
    inp[:, K.I_TAMB] = t_amb
    for k, val in (inputs or {}).items():
        inp[:, getattr(K, k)] = val
    state = K.initial_state(turbine, t_amb)
    state[K.S_OMEGA] = omega0
    out = np.zeros((seconds, K.N_OUTPUTS))
    K.run_kernel(prm, st, vt, np.full(seconds * sub, float(v)), inp, state, out, np.zeros(seconds * sub),
                 np.zeros(seconds * sub))
    return out, state
