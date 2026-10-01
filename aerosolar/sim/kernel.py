"""Fused time-stepping kernel of the turbine physics.

The mechanical state (rotor speed and angle) is integrated at ``physics_hz`` (default 10 Hz); the slow
subsystems (battery, charge controller, thermal nodes, brake supervisor) are updated once per second.

Why semi-implicit integration
-----------------------------
Once the generator is charging, its load torque is steep in speed: dT/dω ≈ ratio² (1.35 K)²/R_eq,
giving a mechanical time constant J/(dT/dω) of only ~0.08 s. Explicit Euler at 10 Hz would ring or
diverge, so we linearise the net torque around the current speed:

    ω_{n+1} = ω_n + Δt T_net(ω_n) / (J - Δt min(0, ∂T_net/∂ω))

which is unconditionally stable for a dissipative load and exact at steady state.

Layout
------
Parameters, per-second inputs, state and per-second outputs are flat float64 arrays indexed by the
named constants below (numba-friendly, zero-copy between Python and the JIT).
"""
from __future__ import annotations

import math

import numpy as np

from ._jit import njit
from .aero import cq_scalar
from .drivetrain import brake_supervisor, running_friction
from .electrical import (CHG_ABSORB, CHG_BULK, CHG_EL_SWEEP, CHG_FLOAT, COULOMBIC_EFF, absorb_acceptance,
                         charge_state_update, generator_load, k_rad_from_mv_per_rpm, ocv, sweep_voltage)
from .thermal import first_order_step, rth_wind

RAD_S_TO_RPM = 60.0 / (2.0 * math.pi)
DERIV_STEP = 1e-3            # finite-difference step for dT/dω [rad/s]
WIND60_TAU_S = 60.0          # time constant of the release-wind average [s]

# ---- parameter indices -------------------------------------------------------------------------
(P_J, P_R, P_A, P_TSR_OPT, P_CP_MAX, P_K_EXP, P_CQ_MIN, P_CONE, P_RIPPLE, P_NBLADES,
 P_RATIO, P_ETA_GEAR, P_BEAR_FRIC, P_BEAR_VISC, P_BEAR_STATIC, P_GREASE_TC, P_COGGING, P_GEN_FRIC,
 P_K_RAD, P_DC_PER_VLL, P_VDIODE, P_REQ, P_FE800,
 P_BRAKE_T, P_OVERSPEED_RPM, P_OVER_PERSIST, P_GUST_MS, P_GUST_PERSIST, P_RELEASE_MS, P_RELEASE_HOLD,
 P_MODE, P_SWEEP_VMIN, P_SWEEP_VMAX, P_SWEEP_PERIOD,
 P_SOC_B2A, P_SOC_A2F, P_SOC_F2B, P_IMAX_CHG, P_RDUMP,
 P_CAP_AH, P_RINT, P_SOC_CUTOFF,
 P_LOAD_W, P_SHED_SOC, P_SHED_W, P_DCDC_EFF,
 P_TB_TAU, P_TB_RTH, P_TB_SOLAR, P_TG_TAU, P_TG_RTH, P_TG_SOLAR, P_CONV_C,
 P_DT, P_SUBSTEPS) = range(55)
N_PARAMS = 55

# ---- per-second input indices ------------------------------------------------------------------
# I_CP_FACTOR multiplies the rotor Cp (natural blade/cone soiling from the solar model)
(I_RHO, I_TAMB, I_GHI, I_WIND_MEAS, I_PPV_AVAIL, I_CP_FACTOR, I_BRAKE_CMD, I_LOAD_EXTRA_W) = range(8)
N_INPUTS = 8

# ---- state indices -------------------------------------------------------------------------------
(S_OMEGA, S_THETA, S_SOC, S_TB, S_TG, S_BRAKE, S_BRAKE_TIMER, S_OVER_CNT, S_GUST_CNT, S_CHG,
 S_VBUS, S_WIND60, S_BMS_CUT, S_TIME, S_TRIP) = range(15)
N_STATE = 15

# ---- per-second output indices -------------------------------------------------------------------
OUTPUT_NAMES = ("omega", "rpm", "tsr", "wind_true", "t_aero", "p_aero", "t_gen_rotor", "i_wind", "v_bus",
                "e_dc", "p_wind_out", "p_dump", "p_pv", "p_load", "soc", "v_batt", "i_batt", "t_bearing",
                "t_gen", "brake", "brake_reason", "chg_state", "bms_cut", "p_fric", "p_brake", "p_gear_loss",
                "p_gen_mech_loss", "p_cu", "p_diode", "ke", "p_bear_heat", "p_gen_heat")
(O_OMEGA, O_RPM, O_TSR, O_WIND, O_T_AERO, O_P_AERO, O_T_GEN, O_I_WIND, O_V_BUS, O_E_DC, O_P_WIND_OUT,
 O_P_DUMP, O_P_PV, O_P_LOAD, O_SOC, O_V_BATT, O_I_BATT, O_TB, O_TG, O_BRAKE, O_BRAKE_REASON, O_CHG,
 O_BMS_CUT, O_P_FRIC, O_P_BRAKE, O_P_GEAR_LOSS, O_P_GEN_MECH, O_P_CU, O_P_DIODE, O_KE, O_P_BEAR_HEAT,
 O_P_GEN_HEAT) = range(32)
N_OUTPUTS = 32


def build_params(turbine: dict) -> np.ndarray:
    """Flatten ``turbine.yaml`` into the kernel parameter vector."""
    r, d, g, b = turbine["rotor"], turbine["drivetrain"], turbine["generator"], turbine["brake"]
    e, th, ctl = turbine["electrical"], turbine["thermal"], turbine["control"]
    cc, bat, sc = e["controller"], e["battery"], e["self_consumption"]
    cones = r.get("cones", {})
    ratio = d["gear_teeth_rotor"] / d["gear_teeth_pinion"]
    p = np.zeros(N_PARAMS)
    p[P_J] = r["inertia_kgm2"] + d["generator_inertia_kgm2"] * ratio ** 2
    p[P_R] = r["diameter_m"] / 2.0
    p[P_A] = r["swept_area_m2"]
    p[P_TSR_OPT], p[P_CP_MAX], p[P_K_EXP], p[P_CQ_MIN] = r["tsr_opt"], r["cp_max"], r["cp_rolloff_exponent"], r["cq_min"]
    p[P_CONE] = cones.get("torque_gain", 1.0) if cones.get("enabled", False) else 1.0
    p[P_RIPPLE], p[P_NBLADES] = r["torque_ripple_2p"], r["blade_count"]
    p[P_RATIO], p[P_ETA_GEAR] = ratio, d["gear_efficiency"]
    p[P_BEAR_FRIC], p[P_BEAR_VISC], p[P_BEAR_STATIC] = d["bearing_friction_nm"], d["bearing_viscous_nms"], d["bearing_static_nm"]
    p[P_GREASE_TC], p[P_COGGING], p[P_GEN_FRIC] = d["grease_temp_coeff_per_k"], d["generator_cogging_nm"], g["friction_loss_nm"]
    p[P_K_RAD] = k_rad_from_mv_per_rpm(g["k_mv_per_rpm"])
    p[P_DC_PER_VLL], p[P_VDIODE], p[P_REQ], p[P_FE800] = g["dc_per_vll"], g["rectifier_drop_v"], g["r_eq_ohm"], g["iron_loss_w_at_800rpm"]
    p[P_BRAKE_T], p[P_OVERSPEED_RPM], p[P_OVER_PERSIST] = b["caliper_torque_nm"], b["overspeed_rpm"], b["overspeed_persist_s"]
    p[P_GUST_MS], p[P_GUST_PERSIST], p[P_RELEASE_MS], p[P_RELEASE_HOLD] = b["gust_trip_ms"], b["gust_persist_s"], b["release_wind_ms"], b["release_hold_s"]
    p[P_MODE] = 1.0 if e.get("mode", "battery") == "eload_sweep" else 0.0
    sw = e.get("eload_sweep", {})
    p[P_SWEEP_VMIN], p[P_SWEEP_VMAX], p[P_SWEEP_PERIOD] = sw.get("v_min", 0.0), sw.get("v_max", 30.0), sw.get("period_s", 600.0)
    p[P_SOC_B2A], p[P_SOC_A2F], p[P_SOC_F2B] = cc["bulk_to_absorb_soc"], cc["absorb_to_float_soc"], cc["float_to_bulk_soc"]
    p[P_IMAX_CHG], p[P_RDUMP] = cc["max_charge_current_a"], cc["dump_resistance_ohm"]
    p[P_CAP_AH], p[P_RINT], p[P_SOC_CUTOFF] = bat["capacity_ah"], bat["r_internal_ohm"], bat["low_voltage_cutoff_soc"]
    p[P_LOAD_W] = sc["pi5_w"] + sc["esp32_w"] + sc["arduino_w"] + sc["sensors_w"] + sc["servo_tracker_idle_w"]
    p[P_SHED_SOC], p[P_SHED_W], p[P_DCDC_EFF] = sc["shed_soc"], sc["shed_power_w"], sc["dcdc_efficiency"]
    p[P_TB_TAU], p[P_TB_RTH], p[P_TB_SOLAR] = th["bearing_tau_s"], th["bearing_rth_k_per_w"], th["bearing_solar_k_per_kwm2"]
    p[P_TG_TAU], p[P_TG_RTH], p[P_TG_SOLAR] = th["generator_tau_s"], th["generator_rth_k_per_w"], th["generator_solar_k_per_kwm2"]
    p[P_CONV_C] = th["convective_wind_coeff"]
    hz = int(ctl.get("physics_hz", 10))
    if hz < 10:
        raise ValueError("physics_hz must be >= 10")
    p[P_DT], p[P_SUBSTEPS] = 1.0 / hz, hz
    return p


def ocv_tables(turbine: dict) -> tuple[np.ndarray, np.ndarray]:
    bat = turbine["electrical"]["battery"]
    return np.asarray(bat["ocv_soc"], dtype=float), np.asarray(bat["ocv_v"], dtype=float)


def initial_state(turbine: dict, t_amb: float) -> np.ndarray:
    """Kernel state at t = 0: rotor at rest, bodies at ambient, battery at its initial SOC."""
    s = np.zeros(N_STATE)
    bat = turbine["electrical"]["battery"]
    s[S_SOC] = bat["soc_initial"]
    s[S_TB] = t_amb
    s[S_TG] = t_amb
    s[S_CHG] = CHG_EL_SWEEP if turbine["electrical"].get("mode") == "eload_sweep" else CHG_BULK
    soc_tab, v_tab = ocv_tables(turbine)
    s[S_VBUS] = float(np.interp(s[S_SOC], soc_tab, v_tab))
    return s


@njit
def _mech_torques(w, v, rho, tb, prm, vbus, dump, cpf, brake_t):
    """Rotor-side torques for omega = w > 0. Returns (T_net, T_aero, T_gen_rotor, T_fric, i, e_dc, p_fe, T_gen_shaft)."""
    vv = max(v, 1e-3)
    tsr = w * prm[P_R] / vv
    t_aero = (0.5 * rho * prm[P_A] * prm[P_R] * vv * vv
              * cq_scalar(tsr, prm[P_TSR_OPT], prm[P_CP_MAX], prm[P_K_EXP], prm[P_CQ_MIN]) * prm[P_CONE] * cpf)
    t_gs, i, e_dc, p_fe = generator_load(w, vbus, dump, prm[P_RATIO], prm[P_K_RAD], prm[P_DC_PER_VLL],
                                         prm[P_VDIODE], prm[P_REQ], prm[P_RDUMP], prm[P_FE800], prm[P_GEN_FRIC])
    t_gen_rotor = prm[P_RATIO] * t_gs / prm[P_ETA_GEAR]
    t_fric = running_friction(w, prm[P_BEAR_FRIC], prm[P_BEAR_VISC], prm[P_GREASE_TC], tb)
    t_net = t_aero - t_gen_rotor - t_fric - brake_t
    return t_net, t_aero, t_gen_rotor, t_fric, i, e_dc, p_fe, t_gs


@njit
def run_kernel(prm, soc_tab, v_tab, wind10, inp, state, out, theta_hi, omega_hi):
    """Advance the turbine by ``inp.shape[0]`` seconds.

    Parameters
    ----------
    prm : parameter vector (see ``P_*``)
    soc_tab, v_tab : battery OCV table
    wind10 : hub wind at the physics rate, length n * substeps [m/s]
    inp : per-second inputs (n, N_INPUTS) (see ``I_*``)
    state : kernel state (N_STATE,), updated in place
    out : per-second outputs (n, N_OUTPUTS), written in place (second means / end values)
    theta_hi, omega_hi : rotor angle [rad, unwrapped] and speed [rad/s] at the end of each sub-step
    """
    n = inp.shape[0]
    sub = int(prm[P_SUBSTEPS])
    dt = prm[P_DT]
    J = prm[P_J]
    ratio = prm[P_RATIO]
    for s in range(n):
        rho = inp[s, I_RHO]
        tb = state[S_TB]
        cpf = inp[s, I_CP_FACTOR]
        eta = prm[P_ETA_GEAR]
        brake_t = prm[P_BRAKE_T] * state[S_BRAKE]
        chg = state[S_CHG]
        if prm[P_MODE] > 0.5:
            vbus = sweep_voltage(state[S_TIME], prm[P_SWEEP_VMIN], prm[P_SWEEP_VMAX], prm[P_SWEEP_PERIOD])
            dump = 0.0
        else:
            vbus = state[S_VBUS]
            dump = 1.0 if chg == CHG_FLOAT else 0.0

        acc_w = 0.0
        acc_tsr = 0.0
        acc_v = 0.0
        acc_taero = 0.0
        acc_paero = 0.0
        acc_tgen = 0.0
        acc_i = 0.0
        acc_e = 0.0
        acc_pfric = 0.0
        acc_pbrake = 0.0
        acc_pgear = 0.0
        acc_pgmech = 0.0
        acc_pcu = 0.0
        acc_pdiode = 0.0
        acc_pout = 0.0
        acc_pbear = 0.0
        for k in range(sub):
            v = wind10[s * sub + k]
            w = state[S_OMEGA]
            if w <= 0.0:
                # Standstill: static aero torque depends on rotor angle (blade-passing ripple). The rotor
                # starts only if it beats breakaway friction + reflected generator cogging + brake.
                vv = max(v, 1e-3)
                t_static = (0.5 * rho * prm[P_A] * prm[P_R] * vv * vv
                            * cq_scalar(0.0, prm[P_TSR_OPT], prm[P_CP_MAX], prm[P_K_EXP], prm[P_CQ_MIN])
                            * prm[P_CONE] * cpf * (1.0 + prm[P_RIPPLE] * math.cos(prm[P_NBLADES] * state[S_THETA])))
                t_break = prm[P_BEAR_STATIC] + ratio * (prm[P_COGGING] + prm[P_GEN_FRIC]) / eta + brake_t
                w_new = 0.0
                if t_static > t_break:
                    t_run = (running_friction(0.0, prm[P_BEAR_FRIC], prm[P_BEAR_VISC], prm[P_GREASE_TC], tb)
                             + ratio * prm[P_GEN_FRIC] / eta + brake_t)
                    w_new = max(0.0, dt * (t_static - t_run) / J)
                t_aero = t_static
                t_gen = 0.0
                t_fric = 0.0
                i = 0.0
                e_dc = 0.0
                p_fe = 0.0
                t_gs = 0.0
            else:
                t_net, t_aero, t_gen, t_fric, i, e_dc, p_fe, t_gs = _mech_torques(
                    w, v, rho, tb, prm, vbus, dump, cpf, brake_t)
                t_net2 = _mech_torques(w + DERIV_STEP, v, rho, tb, prm, vbus, dump, cpf, brake_t)[0]
                dtdw = (t_net2 - t_net) / DERIV_STEP
                w_new = w + dt * t_net / (J - dt * min(dtdw, 0.0))
                if w_new < 0.0:
                    w_new = 0.0
            w_avg = 0.5 * (w + w_new)
            state[S_THETA] += w_avg * dt
            state[S_OMEGA] = w_new
            idx = s * sub + k
            theta_hi[idx] = state[S_THETA]
            omega_hi[idx] = w_new

            # Power bookkeeping (evaluated at the step-average speed) for the energy-balance check.
            wg_avg = ratio * w_avg
            r_eq = prm[P_REQ]
            acc_w += w_new
            acc_v += v
            acc_tsr += w_new * prm[P_R] / max(v, 0.1)
            acc_taero += t_aero
            acc_paero += t_aero * w_avg
            acc_tgen += t_gen
            acc_i += i
            acc_e += e_dc
            acc_pfric += t_fric * w_avg
            acc_pbear += running_friction(w_avg, prm[P_BEAR_FRIC], prm[P_BEAR_VISC], prm[P_GREASE_TC], tb) * w_avg if w_avg > 0 else 0.0
            acc_pbrake += brake_t * w_avg
            acc_pgear += (t_gen - ratio * t_gs) * w_avg
            acc_pgmech += (t_gs - prm[P_DC_PER_VLL] * prm[P_K_RAD] * i) * wg_avg
            acc_pcu += i * i * (r_eq + (prm[P_RDUMP] if dump > 0.5 else 0.0))
            acc_pdiode += i * prm[P_VDIODE]
            acc_pout += i * (vbus if dump < 0.5 else 0.0)

        inv = 1.0 / sub
        w_mean = acc_w * inv
        i_mean = acc_i * inv
        p_out_bus = acc_pout * inv       # power delivered to the bus (battery / e-load)
        p_cu_all = acc_pcu * inv         # includes dump resistor dissipation when dumping
        p_dump = 0.0
        if dump > 0.5:
            p_dump = i_mean * i_mean * prm[P_RDUMP]
        # ---------------------------------------------- battery, PV, loads (1 Hz)
        v_batt_prev = state[S_VBUS] if prm[P_MODE] < 0.5 else ocv(state[S_SOC], soc_tab, v_tab)
        bms_cut = state[S_BMS_CUT]
        if bms_cut > 0.5:
            p_load = 0.0
        elif state[S_SOC] < prm[P_SHED_SOC]:
            p_load = prm[P_SHED_W] / prm[P_DCDC_EFF]
        else:
            p_load = (prm[P_LOAD_W] + inp[s, I_LOAD_EXTRA_W]) / prm[P_DCDC_EFF]
        p_pv = inp[s, I_PPV_AVAIL]
        i_wind_batt = 0.0
        if prm[P_MODE] < 0.5 and dump < 0.5:
            i_wind_batt = p_out_bus / max(vbus, 1.0)
            if chg == CHG_ABSORB:
                acc_i_max = absorb_acceptance(state[S_SOC], prm[P_SOC_B2A], prm[P_SOC_A2F], prm[P_IMAX_CHG])
                if i_wind_batt > acc_i_max:
                    p_dump += (i_wind_batt - acc_i_max) * vbus
                    i_wind_batt = acc_i_max
        if chg == CHG_FLOAT or (chg == CHG_ABSORB and state[S_SOC] > prm[P_SOC_A2F] - 0.005):
            p_pv = min(p_pv, p_load)     # MPPT curtails to the load in float
        i_pv = p_pv / max(v_batt_prev, 1.0)
        i_batt = i_wind_batt + i_pv - p_load / max(v_batt_prev, 1.0)
        cap_as = prm[P_CAP_AH] * 3600.0
        dq = i_batt * (COULOMBIC_EFF if i_batt > 0 else 1.0)
        soc = min(1.0, max(0.0, state[S_SOC] + dq / cap_as))
        state[S_SOC] = soc
        v_batt = ocv(soc, soc_tab, v_tab) + i_batt * prm[P_RINT]
        if prm[P_MODE] < 0.5:
            state[S_VBUS] = v_batt
            state[S_CHG] = charge_state_update(chg, soc, prm[P_SOC_B2A], prm[P_SOC_A2F], prm[P_SOC_F2B])
        if soc < prm[P_SOC_CUTOFF]:
            state[S_BMS_CUT] = 1.0
        elif soc > prm[P_SOC_CUTOFF] + 0.05:
            state[S_BMS_CUT] = 0.0

        # ---------------------------------------------- thermal (1 Hz)
        v_mean = acc_v * inv
        p_bear = acc_pbear * inv
        p_gen_heat = p_cu_all - (p_dump if dump > 0.5 else 0.0) + acc_pgmech * inv
        ghi = inp[s, I_GHI]
        tamb = inp[s, I_TAMB]
        state[S_TB] = first_order_step(state[S_TB], tamb, p_bear, ghi,
                                       rth_wind(prm[P_TB_RTH], prm[P_CONV_C], v_mean), prm[P_TB_SOLAR], prm[P_TB_TAU], 1.0)
        state[S_TG] = first_order_step(state[S_TG], tamb, p_gen_heat, ghi,
                                       rth_wind(prm[P_TG_RTH], prm[P_CONV_C], v_mean), prm[P_TG_SOLAR], prm[P_TG_TAU], 1.0)

        # ---------------------------------------------- brake supervisor (1 Hz)
        wm = inp[s, I_WIND_MEAS]
        a = 1.0 - math.exp(-1.0 / WIND60_TAU_S)
        state[S_WIND60] += a * (wm - state[S_WIND60])
        braked, timer, oc, gc, reason = brake_supervisor(
            state[S_BRAKE], state[S_BRAKE_TIMER], state[S_OVER_CNT], state[S_GUST_CNT], w_mean * RAD_S_TO_RPM,
            wm, state[S_WIND60], inp[s, I_BRAKE_CMD], state[S_BMS_CUT], prm[P_OVERSPEED_RPM], prm[P_OVER_PERSIST],
            prm[P_GUST_MS], prm[P_GUST_PERSIST], prm[P_RELEASE_MS], prm[P_RELEASE_HOLD])
        if braked > 0.5 and state[S_BRAKE] < 0.5:
            state[S_TRIP] = reason
        state[S_BRAKE], state[S_BRAKE_TIMER], state[S_OVER_CNT], state[S_GUST_CNT] = braked, timer, oc, gc
        state[S_TIME] += 1.0

        o = out[s]
        o[O_OMEGA] = w_mean
        o[O_RPM] = w_mean * RAD_S_TO_RPM
        o[O_TSR] = acc_tsr * inv
        o[O_WIND] = v_mean
        o[O_T_AERO] = acc_taero * inv
        o[O_P_AERO] = acc_paero * inv
        o[O_T_GEN] = acc_tgen * inv
        o[O_I_WIND] = i_mean
        o[O_V_BUS] = vbus if dump < 0.5 else i_mean * prm[P_RDUMP]
        o[O_E_DC] = acc_e * inv
        o[O_P_WIND_OUT] = p_out_bus
        o[O_P_DUMP] = p_dump
        o[O_P_PV] = p_pv
        o[O_P_LOAD] = p_load
        o[O_SOC] = soc
        o[O_V_BATT] = v_batt
        o[O_I_BATT] = i_batt
        o[O_TB] = state[S_TB]
        o[O_TG] = state[S_TG]
        o[O_BRAKE] = state[S_BRAKE]
        o[O_BRAKE_REASON] = state[S_TRIP] if state[S_BRAKE] > 0.5 else 0.0
        o[O_CHG] = state[S_CHG] if prm[P_MODE] < 0.5 else CHG_EL_SWEEP
        o[O_BMS_CUT] = state[S_BMS_CUT]
        o[O_P_FRIC] = acc_pfric * inv
        o[O_P_BRAKE] = acc_pbrake * inv
        o[O_P_GEAR_LOSS] = acc_pgear * inv
        o[O_P_GEN_MECH] = acc_pgmech * inv
        o[O_P_CU] = p_cu_all
        o[O_P_DIODE] = acc_pdiode * inv
        o[O_KE] = 0.5 * J * state[S_OMEGA] ** 2
        o[O_P_BEAR_HEAT] = p_bear
        o[O_P_GEN_HEAT] = p_gen_heat


def default_inputs(n: int) -> np.ndarray:
    """Per-second input block with standard air and a clean rotor."""
    inp = np.zeros((n, N_INPUTS))
    inp[:, I_RHO] = 1.225
    inp[:, I_TAMB] = 25.0
    inp[:, I_CP_FACTOR] = 1.0
    return inp
