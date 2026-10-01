"""Electrical system: PMG + rectifier, wind charge controller, LiFePO4 battery, 12 V bus loads.

Generator (DC-side equivalent circuit of a 3-phase PMG into a full-bridge rectifier)
-----------------------------------------------------------------------------------
    n_g   = ratio * n_rotor                         generator speed [rpm]
    E_dc  = 1.35 * K * n_g                          rectified back-EMF [V]  (K: V_LL,rms per rpm)
    I     = max(0, (E_dc - V_d - V_bus) / R_eq)     charging current into the bus [A]
    T_em  = 1.35 * K_rad * I                        electromagnetic torque at generator shaft [N m]

``K_rad`` is K expressed per rad/s. Power balance: E_dc I = V_bus I + V_d I + I^2 R_eq, i.e.
converted power = delivered + diode loss + copper loss. Iron/windage loss scales as n^1.5.

The generator only loads the rotor once E_dc exceeds the battery voltage plus the diode drop, so the
*electrical* cut-in is well above the *rotational* cut-in: with K = 16 mV/rpm the rotor free-wheels
until ~169 rpm (~4 m/s). This is a real consequence of a battery-clamped PMG without MPPT.

Charge controller states: BULK -> ABSORB (SOC >= 0.90) -> FLOAT (SOC >= 0.98; wind diverted to the
dump load to keep the rotor loaded) -> BULK (SOC < 0.95). EL-SWEEP replaces the battery with a
constant-voltage electronic load that sweeps V_bus (for power-curve characterisation).
BMS_CUT: SOC below the low-voltage cut-off disconnects the loads.
"""
from __future__ import annotations

import math

import numpy as np

from ._jit import njit

RPM_PER_RAD_S = 60.0 / (2.0 * math.pi)
COULOMBIC_EFF = 0.99

# Charge-controller state codes (shared with the dashboard)
CHG_BULK = 0.0
CHG_ABSORB = 1.0
CHG_FLOAT = 2.0
CHG_EL_SWEEP = 3.0
CHG_NAMES = {0: "bulk", 1: "absorb", 2: "float/dump", 3: "eload_sweep"}


@njit
def generator_load(omega_rotor: float, v_bus: float, dump: float, ratio: float, k_rad: float,
                   dc_per_vll: float, v_diode: float, r_eq: float, r_dump: float, fe800_w: float,
                   gen_fric: float):
    """Generator reaction at the *generator shaft* and the DC current.

    Parameters
    ----------
    omega_rotor : rotor speed [rad/s]
    v_bus : bus (battery / e-load) voltage [V]
    dump : 1 -> generator feeds the dump resistor r_dump [ohm] instead of the bus
    ratio : gear step-up [-]; k_rad : L-L RMS volts per rad/s at the generator [V s/rad]
    dc_per_vll : rectified DC per V_LL,rms (1.35) ; v_diode : rectifier drop [V]
    r_eq : equivalent DC-side resistance [ohm] ; fe800_w : iron/windage loss at 800 rpm [W]
    gen_fric : Coulomb friction at the generator shaft [N m]

    Returns
    -------
    (T_gen_shaft [N m], I_dc [A], E_dc [V], P_iron [W])
    """
    wg = ratio * omega_rotor
    e_dc = dc_per_vll * k_rad * wg
    e_net = e_dc - v_diode
    i = 0.0
    if dump > 0.5:
        if e_net > 0.0:
            i = e_net / (r_eq + r_dump)
    elif e_net > v_bus:
        i = (e_net - v_bus) / r_eq
    t_em = dc_per_vll * k_rad * i
    w800 = 800.0 / RPM_PER_RAD_S
    p_fe = 0.0
    t_fe = 0.0
    if wg > 1e-9:
        p_fe = fe800_w * (wg / w800) ** 1.5
        t_fe = p_fe / wg
    t_fric = gen_fric if omega_rotor > 1e-9 else 0.0
    return t_em + t_fe + t_fric, i, e_dc, p_fe


@njit
def sweep_voltage(t_s: float, v_min: float, v_max: float, period_s: float) -> float:
    """Sawtooth CV setpoint [V] of the electronic-load sweep."""
    phase = (t_s % period_s) / period_s
    return v_min + (v_max - v_min) * phase


@njit
def ocv(soc: float, soc_tab, v_tab) -> float:
    """LiFePO4 open-circuit voltage [V] from the configured SOC table (linear interpolation)."""
    return np.interp(min(max(soc, 0.0), 1.0), soc_tab, v_tab)


@njit
def charge_state_update(state: float, soc: float, bulk2abs: float, abs2float: float, float2bulk: float) -> float:
    """Charge controller state machine with SOC hysteresis."""
    if state == CHG_BULK and soc >= bulk2abs:
        return CHG_ABSORB
    if state == CHG_ABSORB:
        if soc >= abs2float:
            return CHG_FLOAT
        if soc < bulk2abs - 0.02:
            return CHG_BULK
    if state == CHG_FLOAT and soc < float2bulk:
        return CHG_BULK
    return state


@njit
def absorb_acceptance(soc: float, bulk2abs: float, abs2float: float, i_max: float) -> float:
    """Current the battery accepts in ABSORB [A] (taper model); the rest goes to the dump load."""
    frac = (abs2float - soc) / max(abs2float - bulk2abs, 1e-6)
    return i_max * min(1.0, max(0.1, frac))


def k_rad_from_mv_per_rpm(k_mv_per_rpm: float) -> float:
    """Convert K [mV/rpm, L-L RMS] to [V s/rad]."""
    return k_mv_per_rpm / 1000.0 * RPM_PER_RAD_S
