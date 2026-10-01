# Calibration plan: from twin to prototype

Every model in this repo is parameterised by `config/*.yaml`. When the prototype exists, measure the items below
**in this order**, update the listed config keys, re-run `pytest` and the realism report, then retrain
(`notebooks/05_retrain_on_real_data`).

| # | Measurement | Procedure | Config key(s) updated |
|---|---|---|---|
| 1 | **Sensor offsets** | All 3 DS18B20 and the DHT22 together in a shaded box for 1 h: offsets vs the median. INA219 with zero current (zero offset), then with a known load (multimeter). BH1750 vs a reference lux meter below 50 klx. | `sensors.yaml → ina219.gain_error`, `offset_a`, `bh1750.gain_error` |
| 2 | **Generator constant K** | Drive the generator shaft with a drill at 3–4 speeds (tachometer); measure open-circuit V_LL RMS. Slope = K [mV/rpm]. Then load with a resistor and measure V_dc and I → `r_eq_ohm`. | `generator.k_mv_per_rpm`, `generator.r_eq_ohm`, `generator.rectifier_drop_v` |
| 3 | **Friction & inertia (spin-down)** | Spin the rotor by hand (generator disconnected, then connected, open circuit) and log the Hall edges. Fit `J dω/dt = −(T_c + c_v ω)`. With a known added inertia (a clamped mass), solve for J. | `rotor.inertia_kgm2`, `drivetrain.bearing_friction_nm`, `bearing_viscous_nms`, `generator.iron_loss_w_at_800rpm` |
| 4 | **Breakaway / cut-in** | Torque gauge on the hub: static breakaway torque at 8 angles (this shows the 2P ripple). On the roof, record the lowest wind at which the rotor starts from rest. | `drivetrain.bearing_static_nm`, `generator_cogging_nm`, `rotor.torque_ripple_2p`, `rotor.cut_in_target_ms` |
| 5 | **Cp(TSR) curve** | Wind tunnel (or on-roof, binned by wind): electronic-load sweep (`electrical.mode: eload_sweep`) at fixed wind speeds gives P vs rpm at each wind, hence Cp vs TSR. Include the no-load (runaway) point. Repeat with the cones removed. | `rotor.cp_max`, `tsr_opt`, `cp_rolloff_exponent`, `cq_min`, `cones.torque_gain` |
| 6 | **Thermal time constants** | Step the generator load (resistor on/off) at constant rpm and log T_gen: first-order τ and ΔT/P. Same for the bearing with the rotor spun at constant speed by a motor. | `thermal.*` |
| 7 | **Rooftop wind climate and forecast error** | ≥ 4 weeks of anemometer data: Weibull fit on hourly means, diurnal profile, TI, the ratio to the Open-Meteo 10 m forecast, and the archived Open-Meteo forecasts vs measured wind/irradiance (the real forecast-error statistics). | `site.yaml → wind.*`, `forecast_10m_to_rooftop_ratio`, `simulation.forecast_error.*` |
| 8 | **Battery & loads** | Rest-voltage vs coulomb-counted SOC over one full cycle; INA219 on the 12 V bus for self-consumption. | `electrical.battery.ocv_*`, `r_internal_ohm`, `self_consumption.*` |
| 9 | **Network** | Log packet `seq` gaps and `rx_time − ts` for a week. | `sensors.yaml → network.*` |

After each change, `python -m aerosolar.sim.realism` should still show sensible curves, and the design-table
tests (`tests/test_aero_and_operating_points.py`) should be **updated to the measured values**, not deleted.

For forecasting, the most important steps are 5 (Cp curve), 2 (generator K), 8 (loads and battery: they set PV curtailment) and 7 (real forecast errors).
