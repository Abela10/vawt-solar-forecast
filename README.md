# AeroSolar power forecast

Day-ahead and 5-minute power forecasts for the HCT Dubai hybrid wind–solar turbine: a helical Savonius VAWT plus
an 18 W two-axis tracking PV panel charging a 12.8 V LiFePO4 battery. The turbine is not built yet, so a
physics-based simulator produces exactly what the Raspberry Pi will receive (1 Hz ESP32 packets over MQTT, with
realistic sensor quirks), together with hourly-issued weather forecasts carrying realistic errors.

Scope: [docs/SPEC.md](docs/SPEC.md) · Assumptions and findings: [ASSUMPTIONS.md](ASSUMPTIONS.md) ·
Measurement plan: [CALIBRATION.md](CALIBRATION.md) · Full results: [reports/forecast_evaluation.html](reports/forecast_evaluation.html)

## Results (simulated test year 2026, never used for training, tuning or calibration)

| Statistic | Definition | Value | ≥ 95 % |
|---|---|---|---|
| R², total power, day-ahead | explained variance, hourly | **96.8 %** | yes |
| R², PV power, day-ahead | explained variance, hourly | **97.2 %** | yes |
| R², PV potential, day-ahead | same, hours without curtailment | **97.8 %** | yes |
| R², total potential, day-ahead | same | **97.1 %** | yes |
| Capacity accuracy, total, day-ahead | 1 − RMSE / 30 W | **96.6 %** | yes |
| Capacity accuracy, total potential | 1 − RMSE / 30 W | **96.7 %** | yes |
| Capacity accuracy, PV potential | 1 − RMSE / 18 W | **95.2 %** | yes |
| 95 % interval coverage, total | share of actuals inside the stated 95 % band | **95.1 %** | yes |
| 95 % interval coverage, PV | same | **97.1 %** | yes |
| R², PV, rolling 1–24 h | every hourly issue | **97.2 %** | yes |
| R², total, 5-min nowcast | next-5-min mean power | **99.0 %** | yes |
| Capacity accuracy, 5-min nowcast | 1 − RMSE / 30 W | **98.0 %** | yes |
| Daily energy accuracy, total | mean over days of 1 − \|ΔE\|/E | 94.2 % | no |
| Daily energy accuracy, PV | same | 92.7 % | no |
| Daytime accuracy, total | 1 − Σ\|err\| / Σactual, daytime hours | 91.9 % | no |
| R², wind power, day-ahead | explained variance, hourly | 55.0 % | no |

**Why four stay below 95 %.** The simulated weather forecasts have realistic errors, and these were not reduced
to hit targets. An *oracle* model given the true weather (impossible in practice) reaches only 95.6 % daily energy
accuracy for total and 95.1 % for PV. The production model is within 1.4–2.3 points of that limit. Rooftop wind
power is ~v³, starts only above the ~4.5 m/s electrical cut-in and is cut by the 200 rpm brake, so a day-ahead
wind forecast cannot pin it down hour by hour. Wind is only ~10 % of the energy.

## How it works

* **Inputs**: Open-Meteo-style forecast (10 m wind, shortwave radiation, temperature) with neighbouring-hour,
  daily and cumulative context; clear-sky solar geometry; measurements up to the issue time (last-hour power,
  battery SOC and curtailment state, 72 h PV yield vs clear sky for soiling, 72 h measured/forecast wind ratio
  for NWP bias, same-hour persistence); lead time.
* **Models**: 3-seed LightGBM ensembles per product. PV is learned as a clear-sky index, and night PV is exactly 0.
* **Products**: *delivered* power (what reaches the battery, including PV curtailment when the battery is full,
  which loses 10–11 % of PV energy in spring/summer) and *potential* power (what the panel can produce).
* **Intervals**: conformalized quantile regression per lead block, calibrated on held-out days, so the stated
  80 % / 95 % coverage holds.
* **No leakage**: features use only the forecast issued at the issue time and hours that ended by then
  (`tests/test_forecast.py`; this test caught and fixed a real leak during development).

## Quick start

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt      # Linux/Pi: .venv/bin/python
.venv/Scripts/python -m pip install -e .
.venv/Scripts/python -m pytest

.venv/Scripts/python -m aerosolar.ml.campaign --workers 7    # 2 simulated years of month runs (~15 min)
.venv/Scripts/python -m aerosolar.ml.train --workers 7       # -> models/forecast_<version>.joblib (~15 min)
.venv/Scripts/python -m aerosolar.ml.evaluate                # -> reports/forecast_evaluation.html
.venv/Scripts/python -m aerosolar.tools.run_notebooks        # -> notebooks/*.ipynb, reports/notebooks/*.html
.venv/Scripts/python -m aerosolar.sim.realism                # -> reports/realism.html (simulator sanity check)
```

Live weather input: `aerosolar.ml.forecast_day.fetch_open_meteo(cfg)` returns the current Open-Meteo forecast
in the training schema. **Archive it every hour from day one of the real turbine**; it is the training input
for the real model (see `notebooks/05_retrain_on_real_data`).

## Layout

```
config/          turbine.yaml, site.yaml, sensors.yaml, scenarios.yaml  (every physical number lives here)
aerosolar/
  sim/           weather, aero, drivetrain, electrical, thermal, solar, sensors, kernel (10 Hz physics, numba),
                 engine (streaming), generate, forecasts (synthetic NWP), realism
  ml/            dataset, data_quality, forecast_features, forecast_day, nowcast, metrics, campaign, train, evaluate
  tools/         notebook builder (sources in nb_sources.py)
notebooks/       01-05, executed (HTML in reports/notebooks/)
reports/         forecast_evaluation.html, forecast_metrics.json, realism.html
data/            generated datasets (gitignored)
models/          trained models (binaries gitignored, metadata .json kept)
tests/
```

All numbers come from simulated data. On the real turbine, calibrate the configs (CALIBRATION.md), archive real
forecasts, and retrain.
