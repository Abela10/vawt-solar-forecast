"""Mechanical drivetrain: bearing friction, start-up breakaway, brake supervisor.

Rotor equation of motion (rotor side, omega [rad/s], J [kg m^2] incl. generator inertia x ratio^2):

    J dω/dt = T_aero(v, ω) - T_gen(ω, load)/η_gear - T_fric(ω, T_b) - T_brake          [N m]

Friction model (rotor side):

    T_fric = T_c (1 + c_T max(0, 25 C - T_b)) + c_v ω

T_c: running Coulomb drag of two 6802-2RS (contact seals dominate), c_T: grease viscosity rise when
cold, c_v: viscous drag. From standstill the rotor must also overcome the static breakaway torque
plus the PMG cogging torque reflected through the gear (x ratio). Because the aerodynamic static
torque of a Savonius depends on rotor angle (2P ripple), a stopped rotor may sit at a "dead" angle
and need a gust to start: this gives the start-up hysteresis near cut-in.

Brake supervisor (normally-closed servo disc brake on the drive-gear hub), evaluated at 1 Hz:
    trip if rpm > overspeed for >= persist_s,  or anemometer > gust_ms for >= gust_persist_s,
    or servo power lost (BMS cut-off: the brake is normally closed!).
    release after >= release_hold_s when the 60 s mean wind < release_ms.
"""
from __future__ import annotations

import math

from ._jit import njit

GREASE_REF_C = 25.0


@njit
def running_friction(omega: float, fric_c: float, visc: float, grease_tc: float, t_bearing: float) -> float:
    """Running bearing friction torque [N m] at the rotor for omega > 0 (also the bearing heat source)."""
    cold = max(0.0, GREASE_REF_C - t_bearing)
    return fric_c * (1.0 + grease_tc * cold) + visc * omega


@njit
def brake_supervisor(braked: float, timer: float, over_cnt: float, gust_cnt: float, rpm: float,
                     wind_meas: float, wind60: float, cmd: float, power_lost: float,
                     overspeed_rpm: float, over_persist: float, gust_ms: float, gust_persist: float,
                     release_ms: float, release_hold: float):
    """One 1 Hz update of the brake supervisor.

    Parameters
    ----------
    braked : current brake state (1 engaged / 0 released)
    timer  : seconds since the brake engaged [s]
    over_cnt, gust_cnt : persistence counters [s]
    rpm : rotor speed [rev/min]; wind_meas : anemometer reading [m/s]; wind60 : 60 s mean wind [m/s]
    cmd : manual command (+1 force brake, -1 force release, 0 automatic)
    power_lost : 1 if the servo is unpowered (BMS cut-off) -> normally-closed brake engages

    Returns
    -------
    (braked, timer, over_cnt, gust_cnt, trip_reason) with trip_reason 0 none, 1 overspeed, 2 gust,
    3 manual, 4 power loss.
    """
    over_cnt = over_cnt + 1.0 if rpm > overspeed_rpm else 0.0
    gust_cnt = gust_cnt + 1.0 if wind_meas > gust_ms else 0.0
    reason = 0.0
    if power_lost > 0.5:
        reason = 4.0
    elif cmd > 0.5:
        reason = 3.0
    elif over_cnt >= over_persist:
        reason = 1.0
    elif gust_cnt >= gust_persist:
        reason = 2.0

    if braked > 0.5:
        timer += 1.0
        can_release = (timer >= release_hold and wind60 < release_ms and reason == 0.0) or cmd < -0.5
        if can_release and power_lost < 0.5:
            braked = 0.0
            timer = 0.0
    elif reason > 0.0 and (cmd > -0.5 or reason == 4.0):
        braked = 1.0
        timer = 0.0
    return braked, timer, over_cnt, gust_cnt, reason


def rpm_to_rad_s(rpm: float) -> float:
    return rpm * 2.0 * math.pi / 60.0


def rad_s_to_rpm(omega: float) -> float:
    return omega * 60.0 / (2.0 * math.pi)
