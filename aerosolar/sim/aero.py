"""Rotor aerodynamics of the helical Savonius (drag-type) rotor.

Power-coefficient model
-----------------------
The design gives a single operating point (TSR 0.785, Cp 0.20) which all tabulated wind speeds share.
We need a full Cp(lambda) curve with a realistic roll-off either side. For drag rotors the torque
coefficient falls roughly monotonically from its static value to zero at runaway, so we use

    x      = lambda / lambda_opt
    Cp(x)  = Cp_max * x * ((k + 1) - x^k) / k                                  [-]
    Cq     = Cp / lambda = Cp_max * ((k + 1) - x^k) / (k * lambda_opt)          [-]

Properties: Cp(0) = 0, Cp'(x=1) = 0 and Cp(1) = Cp_max (so the curve *peaks* exactly at the design
point), runaway at x = (k + 1)^(1/k). With k = 2: runaway TSR = 1.36 and static torque coefficient
Cq(0) = 0.38, in the range reported for helical Savonius rotors. Beyond runaway Cq is negative
(the rotor is aerodynamically braked), floored at ``cq_min``.

Torque and power (rho air density [kg/m^3], A swept area [m^2], R radius [m], v wind [m/s]):

    T_aero = 0.5 rho A R v^2 Cq(lambda) * g_cone        [N m]
    P_aero = T_aero * omega = 0.5 rho A v^3 Cp * g_cone [W]
    F_thr  = 0.5 rho A v^2 Ct                           [N]

``g_cone`` is the CFD relative torque gain of the fixed cones (+18 % at 8 m/s), assumed flat.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._jit import njit

R_DRY_AIR = 287.05          # specific gas constant of dry air [J/(kg K)]
R_WATER_VAPOUR = 461.5      # [J/(kg K)]
KELVIN = 273.15
RHO_STANDARD = 1.225        # [kg/m^3], used for the design table


@dataclass(frozen=True)
class AeroParams:
    radius_m: float
    area_m2: float
    tsr_opt: float
    cp_max: float
    k: float
    cq_min: float
    ct: float
    cone_gain: float
    ripple: float
    blades: int

    @classmethod
    def from_config(cls, turbine: dict) -> "AeroParams":
        r = turbine["rotor"]
        cones = r.get("cones", {})
        gain = cones.get("torque_gain", 1.0) if cones.get("enabled", False) else 1.0
        return cls(radius_m=r["diameter_m"] / 2.0, area_m2=r["swept_area_m2"], tsr_opt=r["tsr_opt"],
                   cp_max=r["cp_max"], k=r["cp_rolloff_exponent"], cq_min=r["cq_min"], ct=r["ct_design"],
                   cone_gain=gain, ripple=r["torque_ripple_2p"], blades=int(r["blade_count"]))

    @property
    def runaway_tsr(self) -> float:
        return self.tsr_opt * (self.k + 1.0) ** (1.0 / self.k)

    @property
    def cq_static(self) -> float:
        return self.cp_max * (self.k + 1.0) / (self.k * self.tsr_opt)


def cq_of_tsr(tsr, p: AeroParams):
    """Torque coefficient Cq(lambda) [-] (vectorised). See module docstring."""
    x = np.asarray(tsr, dtype=float) / p.tsr_opt
    cq = p.cp_max * ((p.k + 1.0) - np.abs(x) ** p.k) / (p.k * p.tsr_opt)
    return np.maximum(cq, p.cq_min)


def cp_of_tsr(tsr, p: AeroParams):
    """Power coefficient Cp(lambda) = lambda * Cq(lambda) [-] (without cone gain)."""
    return np.asarray(tsr, dtype=float) * cq_of_tsr(tsr, p)


@njit
def cq_scalar(tsr: float, tsr_opt: float, cp_max: float, k: float, cq_min: float) -> float:
    """Scalar Cq(lambda) for the JIT kernel (same equation as :func:`cq_of_tsr`)."""
    x = abs(tsr) / tsr_opt
    cq = cp_max * ((k + 1.0) - x ** k) / (k * tsr_opt)
    return max(cq, cq_min)


def air_density(t_c, rh_pct=0.0, pressure_hpa: float = 1013.25):
    """Moist-air density [kg/m^3] from temperature [C], relative humidity [%] and pressure [hPa].

    rho = p_d/(R_d T) + p_v/(R_v T), with Tetens saturation pressure. At 45 C the air is ~9 % lighter
    than the 1.225 kg/m^3 design value, which matters for summer power.
    """
    t_k = np.asarray(t_c, dtype=float) + KELVIN
    p_sat = 610.78 * np.exp(17.27 * np.asarray(t_c, dtype=float) / (np.asarray(t_c, dtype=float) + 237.3))
    p_v = np.clip(np.asarray(rh_pct, dtype=float), 0, 100) / 100.0 * p_sat
    p_d = pressure_hpa * 100.0 - p_v
    return p_d / (R_DRY_AIR * t_k) + p_v / (R_WATER_VAPOUR * t_k)


def aero_torque(v, omega, rho, p: AeroParams, cp_factor=1.0):
    """Mean aerodynamic torque [N m] at wind ``v`` [m/s], rotor speed ``omega`` [rad/s]."""
    v = np.maximum(np.asarray(v, dtype=float), 1e-6)
    tsr = np.asarray(omega, dtype=float) * p.radius_m / v
    return 0.5 * rho * p.area_m2 * p.radius_m * v ** 2 * cq_of_tsr(tsr, p) * p.cone_gain * cp_factor


def aero_power(v, omega, rho, p: AeroParams, cp_factor=1.0):
    """Aerodynamic shaft power [W]."""
    return aero_torque(v, omega, rho, p, cp_factor) * np.asarray(omega, dtype=float)


def thrust(v, rho, p: AeroParams):
    """Rotor thrust [N] = 0.5 rho A v^2 Ct (Ct treated as constant for a drag rotor)."""
    return 0.5 * rho * p.area_m2 * np.asarray(v, dtype=float) ** 2 * p.ct
