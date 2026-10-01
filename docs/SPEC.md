# AeroSolar power forecast: specification

> **Scope change (2026-10-01).** The original brief combined a digital twin, predictive maintenance (PdM) and a
> health dashboard. At the owner's request the project now covers **power forecasting only**. All PdM code
> (fault injection, vibration, bearing kinematics, order tracking, anomaly detection, fault classification, health
> index and RUL) was removed. The original combined brief stays in the git history (commit `93b9c10`).

## Goal

Forecast the electrical output of the HCT Dubai hybrid wind–solar turbine (helical Savonius VAWT + 2-axis
tracking PV panel) as accurately as the information available allows, and prove the accuracy honestly.

* **Day-ahead**: hourly PV, wind and total power for the next 24–48 h from a weather forecast (Open-Meteo:
  `wind_speed_10m`, `shortwave_radiation`, `temperature_2m` for 25.1° N 55.4° E), re-issued hourly, with
  calibrated 80 % and 95 % prediction intervals and daily energy (Wh).
* **Nowcast**: mean power over the next 5 minutes from recent measurements.
* Two products: **delivered** power (what reaches the battery, including curtailment when the battery is full)
  and **generation potential** (what the panel/turbine can produce; scored on hours without curtailment).
* Always compared against persistence and climatology baselines.

## Data

There is no measured data yet. A physics-based simulator ("the twin") produces exactly what the Raspberry Pi
will receive: the 19-field ESP32 packets at 1 Hz over Wi-Fi/MQTT with realistic sensor quirks, packet loss and
clock offset, plus hourly-issued synthetic weather forecasts with realistic errors. Switching to the real turbine
means feeding the same packet schema from MQTT and retraining on measured data.

* Physics: weather (Weibull + UAE diurnal sea breeze + shamal + turbulence; solar geometry + dust/cloud),
  rotor aerodynamics (Cp(TSR) through the design table), drivetrain, PMG + rectifier, charge controller with
  dump load, LiFePO4 battery and self-consumption, PV with tracker, temperature and soiling.
* All design values and UNKNOWNs live in `config/*.yaml` and are logged in `ASSUMPTIONS.md`.

## Evaluation rules

* Time-based split: train on one simulated year, calibrate intervals on held-out days of that year, test on a
  second year that is never used for fitting, tuning or calibration.
* No look-ahead: a forecast issued at time *i* uses only the weather forecast issued at *i* and measurements
  from hours that ended by *i* (enforced by a test).
* Every accuracy statistic is explicitly defined (`aerosolar/ml/metrics.py`); weather-forecast errors in the
  simulator are realistic and are **not** reduced to reach targets. An oracle model given the true weather is
  reported as the information ceiling.

## Deliverables

* `aerosolar/` Python package (simulator, data pipeline, models), `config/`, tests.
* `reports/forecast_evaluation.html` + `reports/forecast_metrics.json`.
* Notebooks explaining the forecast (`notebooks/`, executed, HTML in `reports/notebooks/`).
* `ASSUMPTIONS.md`, `CALIBRATION.md`, `README.md`.
