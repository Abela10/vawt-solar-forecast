"""Evaluation of the power-forecast models on the untouched test year → ``reports/forecast_evaluation.html``.

    python -m aerosolar.ml.evaluate

Evaluation sets (all from test year B, never used for fitting, tuning or interval calibration):

* **Day-ahead**: the forecast issued at local midnight (20:00 UTC) for the 24 hours of the coming local day --
  the operational day-ahead product. Daily energy accuracy is computed on this set.
* **Rolling 1-24 h**: every hourly issue, leads 1-24 h (what the dashboard shows, re-issued hourly).
* **Lead curve**: accuracy for every lead 1-48 h.
* **Nowcast**: every minute of the test year, mean power over the next 5 minutes.

Baselines: persistence (same hour on the latest measured day), monthly-hourly climatology, and for the
nowcast the last-minute persistence and the solar "smart persistence" (clear-sky index persistence).
All metric definitions: :mod:`aerosolar.ml.metrics`.
"""
from __future__ import annotations

import html
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..config import REPO_ROOT, Config  # noqa: E402
from ..sim.realism import GRID, INK, INK2, S1, S2, S3, _style  # noqa: E402
from .forecast_day import POINT_PARAMS, PRODUCTS, not_curtailed  # noqa: E402
from .forecast_features import FEATURES, TARGETS  # noqa: E402
from . import campaign as C  # noqa: E402
from .metrics import daily_energy_accuracy, interval_metrics, point_metrics, skill  # noqa: E402
from .train import load_latest, load_split  # noqa: E402

REPORT = REPO_ROOT / "reports"
FIG = REPORT / "forecast"
DAY_AHEAD_ISSUE_UTC_HOUR = 20      # = 00:00 local (UTC+4)
LABELS = {"pv": "PV power", "wind": "Wind power", "total": "Total power",
          "pv_potential": "PV potential", "total_potential": "Total potential"}
DELIVERED = ("pv", "wind", "total")
POTENTIAL = ("pv_potential", "total_potential")


def capacities(cfg: Config) -> dict:
    c = cfg.turbine["forecast"]["capacity_w"]
    cap = {"pv": c["pv"], "wind": c["wind"], "total": c["pv"] + c["wind"]}
    cap.update({"pv_potential": cap["pv"], "total_potential": cap["total"]})
    return cap


def _save(fig, name: str) -> str:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=130)
    plt.close(fig)
    return f"forecast/{name}"


def score_set(df: pd.DataFrame, pred: pd.DataFrame, fda, cap: dict, with_daily: bool,
              products=DELIVERED) -> dict:
    out = {}
    for prod in products:
        t = PRODUCTS[prod][0]
        rows = not_curtailed(df) if PRODUCTS[prod][1] else np.ones(len(df), bool)
        d, pr = df[rows], pred[rows]
        y = d[f"y_{t}"].to_numpy()
        p = pr[f"{prod}_p50"].to_numpy()
        m = point_metrics(y, p, cap[prod])
        pers = d[f"pers_{t}"].to_numpy() if t != "total" else (d["pers_pv"] + d["pers_wind"]).to_numpy()
        clim = fda.climatology_forecast(d, t)
        ok = np.isfinite(y) & np.isfinite(pers) & np.isfinite(clim)
        mae_m = float(np.mean(np.abs(p[ok] - y[ok])))
        m["mae_persistence_w"] = float(np.mean(np.abs(pers[ok] - y[ok])))
        m["mae_climatology_w"] = float(np.mean(np.abs(clim[ok] - y[ok])))
        m["skill_vs_persistence"] = skill(mae_m, m["mae_persistence_w"])
        m["skill_vs_climatology"] = skill(mae_m, m["mae_climatology_w"])
        for k in ("95", "80"):
            iv = interval_metrics(y, pr[f"{prod}_lo{k}"], pr[f"{prod}_hi{k}"])
            m[f"coverage_{k}"], m[f"width_{k}_w"] = iv["coverage"], iv["mean_width_w"]
        if with_daily and not PRODUCTS[prod][1]:
            m.update(daily_energy_accuracy(d[["local_date"]].assign(y=y, p=p), "y", "p"))
        out[prod] = m
    return out


def by_lead(df: pd.DataFrame, pred: pd.DataFrame, cap: dict) -> pd.DataFrame:
    rows = []
    for L, g in df.groupby("lead_h"):
        r = {"lead_h": L}
        for t in TARGETS:
            m = point_metrics(g[f"y_{t}"], pred.loc[g.index, f"{t}_p50"], cap[t])
            r[f"{t}_r2"], r[f"{t}_cap_acc"] = m["r2"], m["cap_accuracy_rmse"]
            pers = g[f"pers_{t}"] if t != "total" else g["pers_pv"] + g["pers_wind"]
            r[f"{t}_pers_cap_acc"] = point_metrics(g[f"y_{t}"], pers, cap[t])["cap_accuracy_rmse"]
        rows.append(r)
    return pd.DataFrame(rows)


def by_month(df: pd.DataFrame, pred: pd.DataFrame, cap: dict) -> pd.DataFrame:
    month = pd.DatetimeIndex(df["valid_time"]).month
    rows = []
    for mo, g in df.groupby(month):
        f = g[["local_date"]].assign(y=g["y_total"], p=pred.loc[g.index, "total_p50"])
        d = daily_energy_accuracy(f, "y", "p")
        rows.append({"month": mo, "daily_energy_accuracy": d["daily_energy_accuracy_mean"],
                     "r2_total": point_metrics(g["y_total"], pred.loc[g.index, "total_p50"], cap["total"])["r2"]})
    return pd.DataFrame(rows)


def nowcast_scores(bundle: dict, cap: dict) -> dict:
    _, nc = load_split("test")
    nc = nc.dropna(subset=["y_pv", "y_wind"])
    pred = bundle["nowcaster"].predict(nc)
    out = {}
    for t in TARGETS:
        y = nc[f"y_{t}"].to_numpy()
        m = point_metrics(y, pred[f"nowcast_{t}"], cap[t])
        pers = nc[f"persist_{t}"].to_numpy()
        ok = np.isfinite(pers)
        m["mae_persistence_w"] = float(np.mean(np.abs(pers[ok] - y[ok])))
        m["skill_vs_persistence"] = skill(float(np.mean(np.abs(pred[f"nowcast_{t}"].to_numpy()[ok] - y[ok]))),
                                          m["mae_persistence_w"])
        if t in ("pv", "total"):
            sp = nc[f"smart_{t}"].to_numpy()
            ok2 = np.isfinite(sp)
            m["mae_smart_persistence_w"] = float(np.mean(np.abs(sp[ok2] - y[ok2])))
            m["skill_vs_smart_persistence"] = skill(float(np.mean(np.abs(pred[f"nowcast_{t}"].to_numpy()[ok2] - y[ok2]))),
                                                    m["mae_smart_persistence_w"])
        out[t] = m
    return out


def oracle_ceiling(train: pd.DataFrame, test: pd.DataFrame, cap: dict) -> dict:
    """Information limit: the same model given the *true* hourly irradiance and wind as extra inputs.

    Fitted on 2025, scored on the 2026 day-ahead set. No real forecaster can know the true weather; the gap between
    this and the production model is the cost of weather-forecast error, the rest is irreducible here.
    Analysis only -- never part of the product.
    """
    import lightgbm as lgb

    def truth(split):
        frames = [pd.read_parquet(C.run_dir(r) / "weather_hourly_truth.parquet").set_index("valid_time")
                  for r in C.runs() if r["split"] == split]
        t = pd.concat(frames)
        return t[~t.index.duplicated()]

    out = {}
    for split, d in (("train", train), ("test", test)):
        tr = truth(split)
        d["ghi_true"] = tr["ghi"].reindex(pd.DatetimeIndex(d["valid_time"])).to_numpy()
        d["wind_true"] = tr["wind_10m_equiv"].reindex(pd.DatetimeIndex(d["valid_time"])).to_numpy()
    feats = FEATURES + ["ghi_true", "wind_true"]
    da = (pd.DatetimeIndex(test["issue_time"]).hour == DAY_AHEAD_ISSUE_UTC_HOUR) & (test["lead_h"] <= 24)
    for t in ("pv", "total"):
        tr = train.dropna(subset=[f"y_{t}"])
        params = {**POINT_PARAMS, "n_estimators": 400}
        m = lgb.LGBMRegressor(random_state=0, verbose=-1, **params).fit(tr[feats], tr[f"y_{t}"])
        te = test[da]
        p = np.maximum(m.predict(te[feats]), 0.0)
        if t == "pv":
            p[te["ghi_clear"].to_numpy() <= 1] = 0.0
        r = point_metrics(te[f"y_{t}"], p, cap[t])
        r.update(daily_energy_accuracy(te[["local_date"]].assign(y=te[f"y_{t}"].to_numpy(), p=p), "y", "p"))
        out[t] = r
    return out


# ------------------------------------------------------------------------------------- figures
def fig_week(df: pd.DataFrame, pred: pd.DataFrame, cfg: Config) -> str:
    """One test week of day-ahead forecasts (midnight issues) vs actuals, with 80 % and 95 % bands."""
    da = df[pd.DatetimeIndex(df["issue_time"]).hour == DAY_AHEAD_ISSUE_UTC_HOUR]
    start = pd.Timestamp(f"{pd.DatetimeIndex(da['valid_time']).year.max()}-03-10", tz="UTC")
    w = da[(da["valid_time"] >= start) & (da["valid_time"] < start + pd.Timedelta(days=7))].sort_values("valid_time")
    p = pred.loc[w.index]
    x = pd.DatetimeIndex(w["valid_time"]).tz_convert(None) + pd.Timedelta(hours=cfg.site["location"]["utc_offset_h"])
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for ax, t, c in zip(axes, ("total", "pv", "wind"), (S1, S2, S3)):
        ax.fill_between(x, p[f"{t}_lo95"], p[f"{t}_hi95"], color=c, alpha=0.12, linewidth=0, label="95 % interval")
        ax.fill_between(x, p[f"{t}_lo80"], p[f"{t}_hi80"], color=c, alpha=0.25, linewidth=0, label="80 % interval")
        ax.plot(x, p[f"{t}_p50"], color=c, linewidth=1.4, label="forecast")
        ax.plot(x, w[f"y_{t}"], color=INK, linewidth=1.0, label="actual")
        ax.set_ylabel(f"{LABELS[t]} [W]")
    axes[0].legend(fontsize=8, ncol=4, loc="upper left")
    axes[0].set_title("Day-ahead forecasts issued at local midnight vs measured (test year, one week)")
    axes[-1].set_xlabel("local time")
    return _save(fig, "week.png")


def fig_lead(lead: pd.DataFrame) -> str:
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for t, c in zip(TARGETS, (S2, S3, S1)):
        ax[0].plot(lead["lead_h"], lead[f"{t}_r2"], color=c, label=LABELS[t])
        ax[1].plot(lead["lead_h"], 100 * lead[f"{t}_cap_acc"], color=c, label=f"{LABELS[t]} model")
        ax[1].plot(lead["lead_h"], 100 * lead[f"{t}_pers_cap_acc"], color=c, linestyle=":", linewidth=1,
                   label=f"{LABELS[t]} persistence")
    ax[0].set_ylabel("R²")
    ax[1].set_ylabel("capacity accuracy 1 − RMSE/C [%]")
    for a in ax:
        a.set_xlabel("lead time [h]")
        a.axhline(0.95 if a is ax[0] else 95, color=INK2, linestyle="--", linewidth=0.8)
    ax[0].legend(fontsize=8)
    ax[1].legend(fontsize=7, ncol=2)
    ax[0].set_title("Explained variance by lead time")
    ax[1].set_title("Capacity accuracy by lead time (dashed: 95 %)")
    return _save(fig, "by_lead.png")


def fig_daily(df: pd.DataFrame, pred: pd.DataFrame) -> str:
    da = df[pd.DatetimeIndex(df["issue_time"]).hour == DAY_AHEAD_ISSUE_UTC_HOUR]
    g = da.assign(p=pred.loc[da.index, "total_p50"]).groupby("local_date")
    full = g.size() >= 22
    e = pd.DataFrame({"y": g["y_total"].sum(), "p": g["p"].sum()})[full].dropna()
    fig, ax = plt.subplots(figsize=(5.4, 5))
    lim = [0, max(e.max()) * 1.05]
    ax.plot(lim, lim, color=INK2, linestyle=":", linewidth=1)
    ax.scatter(e["y"], e["p"], s=12, color=S1, alpha=0.7)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("measured daily energy [Wh]")
    ax.set_ylabel("day-ahead forecast [Wh]")
    ax.set_title("Daily energy, every test day")
    return _save(fig, "daily_energy.png")


def fig_importance(fda) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, t, c in zip(axes, TARGETS, (S2, S3, S1)):
        imp = fda.models[t].importance(10)[::-1]
        ax.barh(imp.index, imp.values * 100, color=c)
        ax.set_title(LABELS[t])
        ax.set_xlabel("share of gain [%]")
        ax.tick_params(axis="y", labelsize=8)
    return _save(fig, "importance.png")


# ------------------------------------------------------------------------------------- report
def _f(x, nd=1, pct=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{100 * x:.{nd}f} %" if pct else f"{x:.{nd}f}"


def _table(rows) -> str:
    out = ["<table>"]
    for i, r in enumerate(rows):
        tag = "th" if i == 0 else "td"
        out.append("<tr>" + "".join(f"<{tag}{' class=num' if j and i else ''}>{html.escape(str(c))}</{tag}>"
                                    for j, c in enumerate(r)) + "</tr>")
    out.append("</table>")
    return "".join(out)


def headline(m: dict) -> list[tuple[str, str, float, bool]]:
    """(statistic, definition, value, >= 95 %) rows for the headline table -- all on the 2026 test year."""
    da, rl, nc = m["day_ahead"], m["rolling_1_24h"], m["nowcast"]
    rows = [
        ("R², total power (day-ahead)", "explained variance, hourly total", da["total"]["r2"]),
        ("R², PV power (day-ahead)", "explained variance, hourly PV", da["pv"]["r2"]),
        ("R², PV potential (day-ahead)", "same, hours without curtailment", da["pv_potential"]["r2"]),
        ("R², total potential (day-ahead)", "same, total", da["total_potential"]["r2"]),
        ("Capacity accuracy, total (day-ahead)", "1 − RMSE / 30 W", da["total"]["cap_accuracy_rmse"]),
        ("Capacity accuracy, total potential (day-ahead)", "1 − RMSE / 30 W, no curtailment", da["total_potential"]["cap_accuracy_rmse"]),
        ("Capacity accuracy, PV potential (day-ahead)", "1 − RMSE / 18 W, no curtailment", da["pv_potential"]["cap_accuracy_rmse"]),
        ("95 % interval coverage, total (day-ahead)", "share of actuals inside the 95 % band", da["total"]["coverage_95"]),
        ("95 % interval coverage, PV (day-ahead)", "same, PV", da["pv"]["coverage_95"]),
        ("R², PV (rolling 1–24 h)", "every hourly issue", rl["pv"]["r2"]),
        ("R², total 5-min nowcast", "next-5-min mean", nc["total"]["r2"]),
        ("Capacity accuracy, total 5-min nowcast", "1 − RMSE / 30 W", nc["total"]["cap_accuracy_rmse"]),
        ("Daily energy accuracy, total (day-ahead)", "mean over days of 1 − |ΔE|/E", da["total"]["daily_energy_accuracy_mean"]),
        ("Daily energy accuracy, PV (day-ahead)", "same, PV only", da["pv"]["daily_energy_accuracy_mean"]),
        ("Daytime accuracy, total (day-ahead)", "1 − Σ|err| / Σactual, hours ≥ 10 % cap", da["total"]["daytime_accuracy"]),
        ("R², wind power (day-ahead)", "explained variance, hourly wind", da["wind"]["r2"]),
    ]
    return [(a, b, c, bool(np.isfinite(c) and c >= 0.95)) for a, b, c in rows]


def write_report(m: dict, meta: dict, imgs: dict, month: pd.DataFrame) -> None:
    css = (REPO_ROOT / "aerosolar" / "sim" / "realism.py").read_text(encoding="utf-8")
    css = css[css.index('css = """') + 9: css.index('"""', css.index('css = """') + 9)]
    css += (".ok{color:#1baf7a;font-weight:600}.no{color:#e34948;font-weight:600}"
            "th{text-align:left;color:var(--ink2);font-weight:600;border-bottom:1px solid var(--rule);padding:6px 8px}")
    hl = headline(m)
    passed = sum(r[3] for r in hl)
    o = m["oracle"]
    da = m["day_ahead"]
    p = [f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
         f"<title>Power Forecast Evaluation</title><style>{css}</style></head><body><main>",
         "<h1>AeroSolar power forecast: evaluation</h1>",
         f"<p class=meta>model {html.escape(meta['version'])} · trained on 2025 (last 5 days of each month held out for "
         f"early stopping and interval calibration) · tested on every hour of {meta['test_year']}, which was never used for "
         "fitting, tuning or calibration</p>",
         "<p><b>All data are simulated.</b> The weather forecasts the model receives carry realistic errors (wind σ ≈ 1.3 m/s, "
         "irradiance σ ≈ 12 % at 24 h). They were <i>not</i> reduced to reach any target: they set the ceiling on what any "
         "model can achieve (see <i>Information limit</i> below). On the real turbine the model must be retrained on measured data.</p>",
         "<h2>Headline statistics (test year)</h2>",
         f"<p>{passed} of {len(hl)} statistics are at or above 95 %. Every one is defined so it can be checked; the ones below "
         "95 % are explained underneath.</p>",
         "<table><tr><th>statistic</th><th>definition</th><th class=num>value</th><th>≥ 95 %</th></tr>"
         + "".join(f"<tr><td>{html.escape(a)}</td><td>{html.escape(b)}</td><td class=num>{_f(c, 1, True)}</td>"
                   f"<td class={'ok' if ok else 'no'}>{'yes' if ok else 'no'}</td></tr>" for a, b, c, ok in hl) + "</table>",
         "<h2>Information limit (why some statistics stay below 95 %)</h2>",
         "<p>The same model was given the <b>true</b> hourly irradiance and wind (impossible in practice) and scored on the same "
         "test days. What it reaches is the ceiling set by everything the weather cannot explain (soiling, tracker error, "
         "battery curtailment, intra-hour turbulence):</p>",
         _table([["", "R²", "capacity accuracy", "daily energy accuracy"],
                 ["PV: forecast model", _f(da["pv"]["r2"], 3), _f(da["pv"]["cap_accuracy_rmse"], 1, True),
                  _f(da["pv"]["daily_energy_accuracy_mean"], 1, True)],
                 ["PV: oracle (true weather)", _f(o["pv"]["r2"], 3), _f(o["pv"]["cap_accuracy_rmse"], 1, True),
                  _f(o["pv"]["daily_energy_accuracy_mean"], 1, True)],
                 ["Total: forecast model", _f(da["total"]["r2"], 3), _f(da["total"]["cap_accuracy_rmse"], 1, True),
                  _f(da["total"]["daily_energy_accuracy_mean"], 1, True)],
                 ["Total: oracle (true weather)", _f(o["total"]["r2"], 3), _f(o["total"]["cap_accuracy_rmse"], 1, True),
                  _f(o["total"]["daily_energy_accuracy_mean"], 1, True)]]),
         "<p><b>Daily energy accuracy</b> is limited by how wrong a day-ahead irradiance forecast is over a whole day, and about "
         "10 % of spring/summer PV energy is curtailed by a full battery (a dispatch decision, not weather). <b>Wind</b> depends on "
         "turbulence and gusts that no weather forecast can know a day ahead; the electrical cut-in (~4.5 m/s) and the 200 rpm "
         "overspeed brake make power very sensitive to small wind errors. Wind is ~10 % of the energy, so the total is dominated by PV.</p>",
         f"<img src='{imgs['week']}' alt='forecast week'>",
         "<h2>Day-ahead (issued at local midnight, next 24 h)</h2>"]
    cols = ["mae_w", "rmse_w", "r2", "cap_accuracy_rmse", "cap_accuracy_mae", "daytime_accuracy", "daily_energy_accuracy_mean",
            "skill_vs_persistence", "skill_vs_climatology", "coverage_80", "coverage_95", "width_95_w"]
    names = ["MAE [W]", "RMSE [W]", "R²", "capacity acc. (RMSE)", "capacity acc. (MAE)", "daytime acc.", "daily energy acc.",
             "skill vs persistence", "skill vs climatology", "80 % coverage", "95 % coverage", "95 % width [W]"]

    def block(d):
        prods = list(d)
        rows = [["", *[LABELS[t] for t in prods]]]
        for c, n in zip(cols, names):
            pct = c not in ("mae_w", "rmse_w", "width_95_w", "r2")
            rows.append([n, *[_f(d[t].get(c, np.nan), 3 if c == "r2" else (1 if pct else 2), pct) for t in prods]])
        return _table(rows)
    p.append(block(da))
    p.append(f"<p>Measured energy per test day: {_f(da['total']['daily_energy_wh_mean'], 1)} Wh "
             f"(PV {_f(da['pv']['daily_energy_wh_mean'], 1)} Wh, wind {_f(da['wind']['daily_energy_wh_mean'], 1)} Wh); "
             f"mean absolute daily error {_f(da['total']['daily_energy_wh_mae'], 2)} Wh over {da['total']['days']} days. "
             "Potential products are scored only on hours without curtailment.</p>")
    p.append(f"<img src='{imgs['daily']}' alt='daily energy' style='max-width:480px'>")
    p.append("<h2>Rolling forecasts (every hourly issue, leads 1–24 h)</h2>")
    p.append(block(m["rolling_1_24h"]))
    p.append(f"<img src='{imgs['lead']}' alt='by lead'>")
    p.append("<h2>By month (day-ahead total)</h2>")
    p.append(_table([["month", "daily energy accuracy", "R²"]]
                    + [[int(r.month), _f(r.daily_energy_accuracy, 1, True), _f(r.r2_total, 3)] for r in month.itertuples()]))
    p.append("<h2>5-minute nowcast</h2>")
    nc = m["nowcast"]
    p.append(_table([["", *[LABELS[t] for t in TARGETS]],
                     ["MAE [W]", *[_f(nc[t]["mae_w"], 3) for t in TARGETS]],
                     ["R²", *[_f(nc[t]["r2"], 3) for t in TARGETS]],
                     ["capacity accuracy (RMSE)", *[_f(nc[t]["cap_accuracy_rmse"], 1, True) for t in TARGETS]],
                     ["skill vs persistence", *[_f(nc[t]["skill_vs_persistence"], 1, True) for t in TARGETS]],
                     ["skill vs smart persistence", *[_f(nc[t].get("skill_vs_smart_persistence", np.nan), 1, True) for t in TARGETS]]]))
    p.append("<h2>What the models rely on</h2>")
    p.append(f"<img src='{imgs['imp']}' alt='feature importance'>")
    p.append("<h2>Method</h2><ul>"
             "<li>Inputs: Open-Meteo-style forecast (10 m wind, shortwave radiation, temperature) with hour-before/after, daily and "
             "cumulative context; clear-sky geometry; observations up to the issue time (last-hour power, battery state, 72 h PV yield "
             "vs clear sky for soiling, 72 h measured/forecast wind ratio for NWP bias, same-hour persistence); lead time.</li>"
             "<li>Models: 3-seed LightGBM ensembles per product, early-stopped on the calibration days; PV learned as a clear-sky "
             "index; night PV fixed at zero.</li>"
             "<li>Intervals: conformalized quantile regression per lead block (1–6, 7–24, 25–48 h), calibrated on held-out 2025 days.</li>"
             "<li>Model choices (features, clear-sky normalisation, potential vs delivered) were made on 2025 data only.</li>"
             "<li>No leakage: features only use the forecast issued at the issue time and hours that ended by then "
             "(<code>tests/test_forecast.py</code>).</li></ul>")
    p.append("</main></body></html>")
    (REPORT / "forecast_evaluation.html").write_text("\n".join(p), encoding="utf-8")


def main() -> None:
    _style()
    cfg = Config.load()
    cap = capacities(cfg)
    bundle, meta = load_latest()
    fda = bundle["day_ahead"]
    test, _ = load_split("test")
    test = test.dropna(subset=["y_total"]).reset_index(drop=True)
    pred = fda.predict(test)
    day_ahead = ((pd.DatetimeIndex(test["issue_time"]).hour == DAY_AHEAD_ISSUE_UTC_HOUR) & (test["lead_h"] <= 24)).to_numpy()
    rolling = (test["lead_h"] <= 24).to_numpy()
    prods = DELIVERED + POTENTIAL
    m = {"model_version": meta["version"], "capacities_w": cap,
         "day_ahead": score_set(test[day_ahead], pred[day_ahead], fda, cap, True, prods),
         "rolling_1_24h": score_set(test[rolling], pred[rolling], fda, cap, False, prods),
         "nowcast": nowcast_scores(bundle, cap)}
    train, _ = load_split("train")
    m["oracle"] = oracle_ceiling(train, test.copy(), cap)
    lead = by_lead(test, pred, cap)
    month = by_month(test[day_ahead], pred[day_ahead], cap)
    imgs = {"week": fig_week(test, pred, cfg), "lead": fig_lead(lead), "daily": fig_daily(test, pred),
            "imp": fig_importance(fda)}
    REPORT.mkdir(exist_ok=True)
    (REPORT / "forecast_metrics.json").write_text(json.dumps(m, indent=2, default=float), encoding="utf-8")
    write_report(m, meta, imgs, month)
    for a, b, c, ok in headline(m):
        print(f"{'PASS' if ok else '    '}  {a:50s} {100 * c:6.2f} %")
    print("oracle daily", {k: round(v["daily_energy_accuracy_mean"], 4) for k, v in m["oracle"].items()},
          "oracle R2", {k: round(v["r2"], 4) for k, v in m["oracle"].items()})
    print(f"wrote {REPORT / 'forecast_evaluation.html'}")


if __name__ == "__main__":
    main()
