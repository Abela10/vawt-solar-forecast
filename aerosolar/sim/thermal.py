"""First-order thermal models for the rotor bearing and the generator.

Each body is a single thermal mass with time constant tau [s] relaxing towards a steady state

    T_ss = T_amb + R_th(v) * P_loss + k_sol * G / 1000                                       [C]
    dT/dt = (T_ss - T) / tau                                                               [C/s]

* bearing : P_loss = bearing friction torque x omega
* generator: P_loss = I^2 R + iron loss + generator friction
* k_sol   : solar heating of the dark mast / housing [K per 1000 W/m^2] -- in UAE summer surfaces pass 70 C
* R_th(v) = R_th0 / (1 + c sqrt(v)) : forced convection lowers the thermal resistance in wind.

These temperatures are part of the telemetry; they also set the grease-viscosity friction term.
"""
from __future__ import annotations

import math

from ._jit import njit

SOLAR_REF_WM2 = 1000.0


@njit
def rth_wind(rth0: float, conv_c: float, wind_ms: float) -> float:
    """Thermal resistance [K/W] under wind [m/s]: R_th0 / (1 + c sqrt(v))."""
    return rth0 / (1.0 + conv_c * math.sqrt(max(wind_ms, 0.0)))


@njit
def first_order_step(t_now: float, t_amb: float, p_loss: float, ghi: float, rth: float, k_solar: float,
                     tau: float, dt: float) -> float:
    """Advance a first-order thermal node by ``dt`` [s] (exact exponential update, unconditionally stable)."""
    t_ss = t_amb + rth * p_loss + k_solar * ghi / SOLAR_REF_WM2
    return t_ss + (t_now - t_ss) * math.exp(-dt / tau)
