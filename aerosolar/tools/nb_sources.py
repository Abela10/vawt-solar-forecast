"""Notebook sources (power forecasting). Each notebook is a list of (kind, text) cells; notebooks only call the package."""

SETUP = """\
%matplotlib inline
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from aerosolar.config import Config, REPO_ROOT
from aerosolar.sim.realism import _style, S1, S2, S3, INK, INK2
_style(); plt.rcParams['figure.dpi'] = 110
cfg = Config.load()
DATA = REPO_ROOT / 'data'
pd.set_option('display.width', 140); pd.set_option('display.max_columns', 30)"""

SETUP_ML = SETUP + """
from aerosolar.ml.train import load_latest, load_split
bundle, meta = load_latest()
print('model version', meta['version'])"""

REFS = """## References
* Romano, Patterson, Candès (2019). *Conformalized quantile regression.* NeurIPS.
* Ke et al. (2017). *LightGBM: a highly efficient gradient boosting decision tree.* NeurIPS.
* Yang et al. (2020). *Verification of deterministic solar forecasts.* Solar Energy 210 (clear-sky index, smart persistence, skill scores).
* Hong et al. (2016). *Probabilistic energy forecasting: GEFCom2014 and beyond.* IJF 32.
* Haurwitz (1945). *Insolation in relation to cloudiness and cloud density.* J. Meteorology.
* Kamoji, Kedare, Prabhu (2009). *Performance tests on helical Savonius rotors.* Renewable Energy 34."""

NB01 = [
    ("md", """# 01 · The turbine, its power curve and its sensors

**Purpose.** Understand what produces the power we forecast: a 0.6 m helical Savonius rotor with flow cones, a 4:1
gear to a small PMG charging a 12.8 V LiFePO4 battery, and an 18 W two-axis tracking PV panel.

```
 wind ─► rotor Cp(λ) ─► gear ─► PMG ─► rectifier ─► charge controller ─► battery ─► 12 V loads (Pi 5, ESP32 …)
  sun ─► PV + tracker ─► MPPT ──────────────────────────────┘   (full battery ⇒ PV curtailed)
```
**Key takeaways**
1. One Cp(λ) curve reproduces the whole design table.
2. The generator charges only above ≈ 4.5 m/s, and the 200 rpm brake trips near 8.2 m/s: wind power is a narrow, steep band.
3. A full battery curtails PV: delivered power depends on the battery, not only on the sun."""),
    ("code", SETUP),
    ("md", """## 1. Rotor aerodynamics

With x = λ/λ_opt: $C_p(x) = C_{p,\\max}\\,x\\,\\frac{(k+1)-x^k}{k}$, peaking at the design point (λ = 0.785, Cp = 0.20).
Torque $T = \\tfrac12\\rho A R v^2 C_q$ [N·m], power $P = \\tfrac12\\rho A v^3 C_p$ [W].

> **Assumption.** The roll-off exponent k = 2 is typical of helical Savonius rotors but unmeasured (CALIBRATION.md step 5)."""),
    ("code", """from aerosolar.sim.aero import AeroParams, cp_of_tsr
p = AeroParams.from_config(cfg.turbine)
tsr = np.linspace(0, 1.5, 300)
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(tsr, cp_of_tsr(tsr, p), color=S1, label='Cp'); ax.plot(tsr, cp_of_tsr(tsr, p) * p.cone_gain, color=S2, label='with cones')
ax.scatter([0.785], [0.2], color=INK, zorder=3, label='design point'); ax.set_xlabel('tip-speed ratio λ'); ax.set_ylabel('Cp'); ax.legend(); plt.show()"""),
    ("md", "**What to notice.** The design point is the peak; torque is highest at standstill (easy start) and vanishes at runaway (λ ≈ 1.36)."),
    ("md", "## 2. The steady-state power curve"),
    ("code", """from aerosolar.sim.realism import model_power_curve
v = np.linspace(0, 11, 45)
pc = model_power_curve(cfg, v)
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(v, pc, color=S1); ax.axvspan(8.2, 11, color=S2, alpha=0.12, label='overspeed brake trips (> 200 rpm)')
ax.set_xlabel('hub wind [m/s]'); ax.set_ylabel('electrical power to battery [W]'); ax.legend(fontsize=8); plt.show()"""),
    ("md", """**What to notice.** Nothing below ≈ 4.5 m/s (the PMG voltage must exceed the battery), a cubic rise, and in real
operation the brake above ≈ 8.2 m/s. A 1 m/s wind-forecast error near 6 m/s changes power by ~50 %: this is why wind is
the hard part of the forecast.

> **Caution (design finding).** Both limits follow from the design numbers (K, battery voltage, trip setting); see ASSUMPTIONS.md."""),
    ("md", "## 3. What the Pi measures\nThe forecast is trained on the INA219 readings (wind and PV power), the anemometer, the BH1750 and the battery state — exactly the 19-field ESP32 packet:"),
    ("code", """from aerosolar.sim.engine import Engine
r = Engine(cfg, {'start': '2026-04-12T06:00:00'}, seed=3).step(4 * 3600)
r.packets.drop_duplicates('seq').head(3).T"""),
    ("md", """**What to notice.** Quirks the cleaning step handles: 0.667 m/s anemometer steps, ±60 rpm Hall steps, BH1750 saturation at
65 535 lx (≈ 546 W/m²), DS18B20 glitch codes, Wi-Fi losses and an ESP32 clock offset.

## What we learned
* The power curve is narrow and steep for wind, smooth and predictable for PV.
* Delivered PV depends on the battery (curtailment), which the forecast must model.

## What changes with real data
Measure the Cp curve, K and the loads (CALIBRATION.md); the forecast is then retrained on the measured stream.

""" + REFS),
]

NB02 = [
    ("md", """# 02 · Simulated data and the forecasting dataset

**Purpose.** Show how two years of realistic data were produced, and how they become the forecasting table.

```
 2025: 12 month runs (train; last 5 days/month = calibration)   2026: 12 month runs (test, untouched)
 each run: 1 Hz packets + hourly-issued 48 h weather forecasts ─► clean ─► hourly actuals ─► feature table
```
**Key takeaways**
1. The weather forecasts carry realistic errors that grow with lead time; they were not tuned.
2. Features use only information available at the issue time (tested).
3. Spring/summer curtailment removes ~10 % of PV energy."""),
    ("code", SETUP),
    ("code", """from aerosolar.ml import campaign as C
pd.DataFrame(C.runs())[['split', 'year', 'month', 'seed', 'hours']].pivot(index='month', columns='split', values='seed')"""),
    ("md", "## 1. Weather-forecast error vs lead time\nThe forecast the model receives, compared with what actually happened (one test month):"),
    ("code", """run = C.run_dir([r for r in C.runs() if r['split'] == 'test'][5])
fc = pd.read_parquet(run / 'forecast_issued.parquet'); tr = pd.read_parquet(run / 'weather_hourly_truth.parquet').set_index('valid_time')
fc['wind_err'] = fc['wind_speed_10m'] - tr['wind_10m_equiv'].reindex(fc['valid_time']).to_numpy()
fc['ghi_err'] = fc['shortwave_radiation'] - tr['ghi'].reindex(fc['valid_time']).to_numpy()
g = fc.groupby('lead_h')
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
ax[0].plot(g['wind_err'].std(), color=S1); ax[0].set_ylabel('wind error σ [m/s]'); ax[0].set_xlabel('lead [h]')
day = fc[tr['ghi'].reindex(fc['valid_time']).to_numpy() > 100]
ax[1].plot(day.groupby('lead_h')['ghi_err'].std(), color=S2); ax[1].set_ylabel('irradiance error σ [W/m²]'); ax[1].set_xlabel('lead [h]')
plt.show()"""),
    ("md", "**What to notice.** Errors grow with lead time, as real NWP errors do. These set the ceiling for any forecaster."),
    ("md", "## 2. Curtailment"),
    ("code", """rows = []
for r in C.runs():
    if r['split'] != 'train': continue
    h = pd.read_parquet(C.run_dir(r) / 'derived' / 'hourly.parquet')
    day = h['ghi_clear'] > 50
    rows.append({'month': r['month'], 'daylight hours with PV curtailed %': 100 * h.loc[day, 'float_frac'].mean(),
                 'mean battery SOC %': h['soc'].mean()})
pd.DataFrame(rows).set_index('month').round(1)"""),
    ("md", """**What to notice.** In spring and summer the battery fills by early afternoon and the controller throttles the panel; in those months this costs 10–11 % of the available PV energy (measured on the simulator's internal record).

> **Assumption.** 5.7 W self-consumption and a 12 Ah battery; both decide how often this happens (CALIBRATION.md step 8).

## 3. The feature table"""),
    ("code", """from aerosolar.ml.forecast_features import FEATURES
from aerosolar.ml.train import load_split
tab, _ = load_split('test')
print(len(tab), 'rows ·', len(FEATURES), 'features')
tab[['issue_time', 'valid_time', 'lead_h', 'sw', 'ws10', 'kt_fc', 'obs_soc', 'pv_yield_72h', 'sw_cum', 'y_pv', 'y_wind']].head(8)"""),
    ("md", """**What to notice.** One row per (issue time, valid hour). Observation features (`obs_*`, `pv_yield_72h`, persistence) only use
hours that ended by the issue time; `tests/test_forecast.py` corrupts all later data and checks the features do not change.
That test caught a real leak during development (48 h persistence).

## What we learned
* A fair forecasting dataset needs realistic forecast errors and strict time ordering.

## What changes with real data
Archive the real Open-Meteo forecasts every hour from day one: they become the training input.

""" + REFS),
]

NB03 = [
    ("md", """# 03 · The day-ahead forecast and how accurate it is

**Purpose.** Train-free walkthrough of the production model and its honest evaluation on the untouched 2026 year.

```
 feature table ─► 3-seed LightGBM ensemble (PV as clear-sky index) ─► P50
               └► quantile LightGBMs ─► conformal widening (held-out 2025 days) ─► 80 % / 95 % bands
```
CQR: with raw band $[q_{lo}, q_{hi}]$, conformity score $E = \\max(q_{lo}-y,\\ y-q_{hi})$ on the calibration set; the band is widened by
its $(1-\\alpha)(1+1/n)$ quantile, which guarantees the stated coverage on exchangeable new data.

**Key takeaways**
1. Hourly PV and total power are explained to R² ≈ 0.97; capacity accuracy ≥ 95 %.
2. Calibrated 95 % intervals cover ~95 % of the test hours.
3. Daily energy and wind are below 95 %: the oracle ceiling shows why."""),
    ("code", SETUP_ML),
    ("code", """import json
from aerosolar.ml.evaluate import headline
m = json.loads((REPO_ROOT / 'reports' / 'forecast_metrics.json').read_text())
pd.DataFrame([(a, b, f'{100 * c:.1f} %', 'yes' if ok else 'no') for a, b, c, ok in headline(m)],
             columns=['statistic', 'definition', 'value (2026 test year)', '≥ 95 %'])"""),
    ("code", """from IPython.display import Image, display
for f in ('week.png', 'by_lead.png', 'daily_energy.png', 'importance.png'):
    display(Image(filename=str(REPO_ROOT / 'reports' / 'forecast' / f)))"""),
    ("md", """**What to notice.**
* The week plot: forecasts follow the daily solar cycle and cloudy/dusty days; actual wind power is spiky and the band is wide there.
* Accuracy by lead: the first hours benefit from the latest observations, then accuracy settles.
* Daily energy: points hug the diagonal; the misses are days with large irradiance-forecast errors.
* The model relies most on forecast irradiance, clear-sky geometry and recent PV yield (soiling)."""),
    ("code", """o = m['oracle']; da = m['day_ahead']
pd.DataFrame({'forecast model': [da['pv']['r2'], da['pv']['daily_energy_accuracy_mean'], da['total']['daily_energy_accuracy_mean']],
              'oracle (true weather)': [o['pv']['r2'], o['pv']['daily_energy_accuracy_mean'], o['total']['daily_energy_accuracy_mean']]},
             index=['PV R²', 'PV daily energy accuracy', 'total daily energy accuracy']).round(3)"""),
    ("md", """**What to notice.** Even a model that knows the true weather cannot get far above 95 % daily energy accuracy here: soiling,
tracker error and curtailment are not in the weather. The remaining gap is the day-ahead forecast error itself.

> **Caution.** All numbers are from simulated data with assumed forecast errors. Real accuracy is known only after weeks
> of real data (notebook 05).

## What we learned
* Gradient boosting with physically meaningful features (clear-sky index, curtailment context, soiling tracker) reaches the
  information limit for PV; wind remains limited by weather predictability.

""" + REFS),
]

NB04 = [
    ("md", """# 04 · 5-minute nowcast

**Purpose.** Short-term power for the dashboard and for battery/load decisions: the mean power over the next 5 minutes.

```
 last 15 min of cleaned 1-min means (power, wind, irradiance) + known future clear-sky ─► LightGBM ─► next 5 min
```
Baselines: persistence (last minute) and, for PV, *smart persistence* — the last 15 min clear-sky index times the
clear-sky irradiance 5 min ahead, the standard solar nowcast reference."""),
    ("code", SETUP_ML),
    ("code", """import json
m = json.loads((REPO_ROOT / 'reports' / 'forecast_metrics.json').read_text())['nowcast']
pd.DataFrame(m).T[['mae_w', 'r2', 'cap_accuracy_rmse', 'skill_vs_persistence', 'skill_vs_smart_persistence']].round(3)"""),
    ("code", """_, nc = load_split('test')
d = nc[(nc['ts'] >= pd.Timestamp('2026-04-15', tz='UTC')) & (nc['ts'] < pd.Timestamp('2026-04-16', tz='UTC'))].dropna(subset=['y_total'])
p = bundle['nowcaster'].predict(d)
x = d['ts'].dt.tz_convert(None) + pd.Timedelta(hours=4)
fig, ax = plt.subplots(figsize=(10, 3.4))
ax.plot(x, d['y_total'], color=INK, lw=0.8, label='actual (next 5 min mean)'); ax.plot(x, p['nowcast_total'], color=S1, lw=0.8, label='nowcast')
ax.set_ylabel('total power [W]'); ax.legend(fontsize=8); plt.show()"""),
    ("md", """**What to notice.** The nowcast tracks the day closely (R² ≈ 0.99 for total power). Its skill over persistence is
modest because 5 minutes is short: most of the value is in getting the sunrise/sunset ramps and gust periods right.

""" + REFS),
]

NB05 = [
    ("md", """# 05 · Retraining on real data

**Purpose.** The recipe for moving from the twin to the prototype: same code, real data.

```
 MQTT vawt/sensors ─► store ─► export month folders in the dataset layout ─► python -m aerosolar.ml.train ─► evaluate
 Open-Meteo every hour ─► archive ─► forecast_issued.parquet
```
**Steps**
1. **From day one, archive the Open-Meteo forecast every hour** (`aerosolar.ml.forecast_day.fetch_open_meteo`). Without the
   forecasts that were *issued*, no fair day-ahead model can be trained.
2. Calibrate the configs (CALIBRATION.md), so the twin matches the hardware.
3. Export each month to `data/<name>/` with `telemetry_1hz.parquet`, `status_1hz.parquet`, `forecast_issued.parquet`, `meta.json`.
4. Point `aerosolar/ml/campaign.py` at the real folders: earliest months = train (keep the last 5 days of each month for
   calibration), the latest month(s) = test. Never evaluate on months used for fitting.
5. Until a full year exists, train on *simulated year + real months* and weight the real months up; the twin then fills the
   seasons not yet seen.
6. Accept the model when the real test month meets the same statistics as in `reports/forecast_evaluation.html`."""),
    ("code", SETUP),
    ("code", """from aerosolar.ml.forecast_day import fetch_open_meteo
try:
    fc = fetch_open_meteo(cfg); print(len(fc), 'hours fetched'); display(fc.head())
except Exception as e:
    print('Open-Meteo not reachable from here:', type(e).__name__)"""),
    ("md", """**What to notice.** The live forecast arrives in exactly the schema the model was trained on, so the switch is a data
source change, not a code change.

> **Caution.** Real forecast errors will differ from the simulated ones; re-run the evaluation on real months before trusting
> the stated accuracy.

""" + REFS),
]
