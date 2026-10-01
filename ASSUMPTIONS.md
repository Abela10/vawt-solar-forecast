# Assumptions register

The turbine has not been built yet, so **no value in this repository is measured**. This file lists every value
that is UNKNOWN in the design documents, or that the model had to invent. For each one it gives the value used,
why it was chosen, and how to measure it on the real prototype. The config keys are in `config/*.yaml`.
[CALIBRATION.md](CALIBRATION.md) turns these into a measurement plan.

Legend: **U** = marked UNKNOWN in the original brief · **M** = modelling assumption added by the twin · **H** = hardware action item.

Scope: power forecasting only (the predictive-maintenance part of the original brief was removed on 2026-10-01).

## Rotor & aerodynamics (`config/turbine.yaml` → `rotor`)

| Key | Value | Kind | Why | How to measure |
|---|---|---|---|---|
| `blade_count` | 2 | U | Spec default (report also says 4) | Count on the CAD / printed rotor |
| `twist_deg` | 180 | U | "~180°" in spec | CAD |
| `overlap_ratio` | 0.0 | U | Spec default; typical 0.15–0.20 | CAD; the Cp curve changes a lot with overlap |
| `inertia_kgm2` | 0.020 | U | 2-blade estimate, bounds 0.011–0.025 | SolidWorks Lzz, or a spin-down test (see CALIBRATION) |
| `cp_max`, `tsr_opt` | 0.20 @ 0.785 | Design | Design point | Wind-tunnel or on-roof power curve |
| `cp_rolloff_exponent` | 2 | M | Gives runaway TSR 1.36 and static Cq 0.38, typical of helical Savonius | Free-running (no-load) rpm vs wind gives the runaway TSR; the stall torque gives Cq(0) |
| `cq_min` | −0.15 | M | Aerodynamic braking beyond runaway | Overspeed coast-down |
| `torque_ripple_2p` | 0.10 | U/M | The helix reduces ripple; 180° twist is nearly smooth | Static torque vs angle (torque wrench, locked rotor) |
| `cones.torque_gain` | 1.18, flat vs wind | M | CFD relative gain at 8 m/s only; assumed the same at all speeds | Power curve with and without cones |
| `joints.imbalance_1p_g/2p_g` | 0.002 / 0.001 g | U | Effect of the 3 bonded print segments | 1P/2P amplitude on the first spin-up; balance and re-measure |
| `ct_design` | 1.1, constant | Design | Drag rotor, Ct ≈ constant | Load cell on the mast |

**Cp curve form** (M): `Cp = Cp_max·x·((k+1) − x^k)/k`, with `x = TSR/TSR_opt`. It passes exactly through all the
tabulated operating points, which all share TSR 0.785. It **peaks** at the design point. If measurements show the
design point is not the peak, change `tsr_opt`/`cp_max` and refit `k`.

## Drivetrain, generator, brake

| Key | Value | Kind | Why | How to measure |
|---|---|---|---|---|
| `generator.k_mv_per_rpm` | 16.0 (band 14–20) | U | Generator not purchased yet | Open-circuit V_LL vs rpm (drill drive) |
| `generator.r_eq_ohm` | 2.8 Ω | M | Chosen so 8 m/s with cones gives 200 rpm / 0.89 N·m / 18.7 W, matching the design table | Loaded test: V_dc and I at known rpm |
| `generator.iron_loss_w_at_800rpm` | 1.5 W | M | Typical small PMG | No-load spin-down with and without the generator coupled |
| `generator.rectifier_drop_v` | 1.4 V | M | Two silicon diodes | Datasheet; Schottky bridge ≈ 0.8 V |
| `drivetrain.gear_efficiency` | 0.97 | M | Healthy spur mesh | Torque in vs out on the bench |
| `drivetrain.bearing_friction_nm` | 0.015 | M | 2× 6802-2RS, contact seals dominate | Spin-down test (see CALIBRATION) |
| `drivetrain.bearing_static_nm`, `generator_cogging_nm` | 0.025, 0.015 | M | Sets the rotational cut-in (~2.2 m/s) | Breakaway torque with a torque gauge |
| `brake.gust_trip_ms` | 12 | U | Spec says 12 or 15 | Design decision |
| `brake.overspeed_persist_s`, `release_*` | 2 s, 9 m/s, 300 s | M | Debounce and hysteresis were not specified | Tune in firmware |

### Findings the twin exposes (please review with the team)

1. **The electrical cut-in is ~4.5–5 m/s, not ≤ 3 m/s.** The PMG is clamped by the 12.8 V battery (no wind MPPT).
   It only delivers current once `1.35·K·n_gen − V_diode > V_batt`. With K = 16 mV/rpm that needs about 169 rotor rpm.
   Below that the rotor free-wheels near runaway and produces nothing. The rotational cut-in (~2.2 m/s) does meet
   the ≤ 3 m/s target. Options: a higher K, a boost converter or MPPT on the wind input, or a 6.4 V battery bank.
2. **The 200 rpm overspeed trip is reached at ~8.2 m/s.** 200 rpm is the *design* operating speed at 8 m/s, so
   gusts in any hour with a 6–9 m/s mean trip the brake. In the first 30-day healthy run this happened ~33 times a
   day, every trip an overspeed trip (each is a ≥ 300 s stop plus brake-pad wear). Consider a 230–250 rpm trip, and
   speed regulation by the dump load before braking.
2b. **The 2 N·m calliper cannot hold a parked rotor in strong wind.** Static aero torque with cones is
   ≈ 0.021·v² N·m (Cq(0) = 0.38 × 1.18), which reaches 2 N·m at ~9.8 m/s and ~3 N·m at 12 m/s. The 2 N·m sizing
   covers *stopping from 200 rpm at 8 m/s*, not *holding at the 12–15 m/s gust trip*. In the twin a braked rotor
   creeps round in shamal gusts. Size the calliper for holding (≥ 5 N·m at 15 m/s), or add a parking lock.
3. **The chain efficiency emerges as 63–65 % at 6–8 m/s**, against the assumed 60 %. `target_efficiency` is
   documentation only and is not enforced.
4. **Self-consumption is comparable to production.** Pi 5 + ESP32 + Arduino + sensors + servo draw ≈ 5.7 W from the
   battery, about 135 Wh/day. At a rooftop Weibull scale of 4.5 m/s, wind contributes little and PV must cover the
   load. The normally-closed brake **engages when the BMS cuts the load** (the servo loses power). The twin models
   this, and it means wind cannot recharge a flat battery. Flag for the hardware design.
5. **BH1750 saturates** at 65535 lx ≈ 546 W/m² (at /120), so it is blind at UAE noon. Use a neutral-density filter,
   or a pyranometer / reference cell.
6. **Anemometer debounce:** a 50 ms lockout caps the count at 20 pulses/s (≈ 13.3 m/s). Above that a strict lockout
   drops pulses, and the reading can *fall*, which defeats the 12–15 m/s gust trip. Use a shorter debounce or a
   hardware RC filter plus a period measurement.

## Electrical system

| Key | Value | Kind | Why |
|---|---|---|---|
| `battery.ocv_*` table, `r_internal_ohm` 0.04 Ω | LiFePO4 4S typical | M | Measure OCV at rest vs coulomb count |
| `controller.*` thresholds, `dump_resistance_ohm` 8 Ω | M | Controller not selected |
| `self_consumption.*` (Pi 3.5 W, ESP32 0.5, Arduino 0.3, sensors 0.3, servo 0.4, DC-DC 88 %) | M | Measure with the INA219 on the 12 V bus |
| `pv.derate` 0.90 | Design | Reproduces the 16.2 W PV value in the design table |
| `pv.temp_coeff_per_k` −0.4 %/K, Faiman u0/u1 | M | Datasheet / field |

## Site & weather (`config/site.yaml`)

| Key | Value | Kind | Why | How to measure |
|---|---|---|---|---|
| `wind.weibull_scale_ms` | 4.5 | U | Earlier modelling used 6.0, likely high for the rooftop | ≥ 1 month of on-roof anemometer data |
| `wind.max_hourly_mean_ms` | 12 (soft tanh cap) | M | Without it, Weibull tail × afternoon peak × shamal gave unphysical 39 m/s hourly means | Local met-station maxima |
| `wind.diurnal_profile` | sea-breeze shape | M | UAE coastal climatology (qualitative) | Same data, hour-of-day means |
| `wind.turbulence_intensity` | 0.22 | M | Rooftop TI 15–30 % | 1 Hz anemometer std/mean over 10 min (corrected for cup inertia) |
| `wind.integral_length_m` | 25 | M | Roof-generated turbulence | Autocorrelation of the wind signal |
| `wind.shamal.*` | 1.5 events/30 d, ×1.5–2.1, 1–3 d | M | Qualitative | Local met records |
| `wind.forecast_10m_to_rooftop_ratio` | 0.80 | M | Roof sheltering vs NWP 10 m | Regress measured hub wind on Open-Meteo |
| `mounting.stand_height_m` | 1.5 | U | Spec default | As built |
| `mounting.reference_height_m` | 2.0 | M | Height the Weibull scale refers to | — |
| `mounting.roof_shear_alpha` | 0.25 | M | Near-roof shear | Two anemometer heights |
| `solar.soiling.base_rate_per_day` | 0.3 %/day | M | UAE literature range 0.2–0.5 %/day | PV output before/after cleaning |
| `solar.soiling.cleaning_interval_days` | 14 | M | Assumed maintenance routine | — |
| `climate.*` | Dubai monthly normals (approx.) | M | Climatology | — |

**Weibull semantics** (M): the configured Weibull describes the *hourly-mean* hub wind before the diurnal and
shamal multipliers. The 10-minute distribution that results is somewhat different (see `reports/realism.html`).

## Sensors & data path (`config/sensors.yaml`)

| Item | Assumption | Kind |
|---|---|---|
| **Packet schema** (19 fields) | `ts, seq, wind_ms, rpm, hall_count, bus_v, current_a, power_w, pv_power_w, batt_v, soc_pct, t_gen_c, t_bearing_c, t_amb_c, lux, ldr_ew, ldr_ns, dht_t_c, dht_rh` | M: the spec gives the count, not the list. PV power needs a second INA219 on the MPPT output, and SOC comes from a BMS/coulomb counter. Both are assumed. |
| INA219 location | Rectified wind output (before the charge controller) | M |
| Controller status | Brake, charge state and load shedding are published by the Pi on `vawt/status` (not by the ESP32) | M |
| Hall rpm field | `rpm_method: count`: 60 × pulses in the last 1 s (±60 rpm steps) | Spec; `period` mode available |
| DS18B20 ambient | Unshielded; reads +2 K per 1000 W/m² in sun | M: fit a radiation shield |
| DS18B20 glitch rates | 85 °C: 2e-5/read; −127 bursts: 5e-6/read, half of them bus-wide | M |
| Network | Gilbert-Elliott loss (~0.3 % average, in bursts), 0.1 % duplicates, 0.2 % late | M |
| ESP32 clock | 0.35 s offset at boot, 30 ppm drift, hourly NTP resync (±20 ms) | M |

## Power-forecast pipeline (`aerosolar/ml/`): choices and findings

| Item | Choice / finding | Kind |
|---|---|---|
| Data | Two simulated years of month-long runs: 2025 = training (last 5 days of each month held out for early stopping and interval calibration), 2026 = test, never used for fitting, tuning or calibration. Every month is an independent weather realisation. | M |
| Weather-forecast error | Synthetic Open-Meteo forecasts: wind σ 1.3 m/s and irradiance σ 12 % at 24 h lead, growing with √lead, a +0.3 m/s roof-level wind bias, errors correlated in time and between successive issues (`site.yaml → simulation.forecast_error`). **These set the accuracy ceiling and were not tuned to reach any target.** | M |
| Rooftop correction | NWP 10 m wind ≈ rooftop hub wind / 0.80 (`forecast_10m_to_rooftop_ratio`); the model learns this from data. | M |
| Forecast capacities | PV 18 W (nameplate); wind 12 W = electrical output at the 200 rpm overspeed limit. The 30 W generator rating is never reachable, so normalising by it would flatter the model (`turbine.yaml → forecast.capacity_w`). | M |
| Negative power readings | Hourly measured power is clipped at 0 W: small negatives are the INA219 zero offset, not energy. | M |
| **Finding: PV curtailment** | With a 12 Ah battery and ~5.7 W self-consumption, the battery is full on many spring/summer afternoons and the controller curtails PV: **10–11 % of the available PV energy is lost** in those months. Delivered PV therefore depends on battery state, not only on the sun. The model forecasts both *delivered* and *potential* power. A larger battery or a useful dump load (e.g. water heating) would recover this energy. | finding |
| **Finding: wind forecastability** | Hour-by-hour rooftop wind power is hard to forecast a day ahead (R² ≈ 0.5): power is ~v³, starts only above the ~4.5 m/s electrical cut-in and is cut by the 200 rpm overspeed brake, so small wind-forecast errors become large power errors. Wind is ~10 % of the energy. | finding |
| Intervals | Conformalized quantile regression per lead block, calibrated on held-out 2025 days. | M |

## Software environment

* Python **3.10** is used locally; the spec asks for 3.11. The code avoids 3.11-only features and runs on both.
  The Pi 5 image (Bookworm) ships Python 3.11.
* The physics kernel uses **numba** when available, with a pure-Python fallback. Behaviour is identical and only
  speed differs.
