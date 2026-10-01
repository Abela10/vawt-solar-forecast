"""Realism check: figures and tables showing the simulated data behaves like a real rooftop turbine.

Usage::

    python -m aerosolar.sim.realism --run data/fc_2026_06_s206

Writes ``reports/realism/*.png`` and ``reports/realism.html``.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from ..config import REPO_ROOT, Config  # noqa: E402
from . import kernel as K  # noqa: E402
from .weather import weibull_fit  # noqa: E402

# Reference categorical palette (light surface) + text tokens
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
REPORT_DIR = REPO_ROOT / "reports"
FIG_DIR = REPORT_DIR / "realism"
DS_GLITCH = (85.0, -127.0)


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
        "axes.titleweight": "semibold", "legend.frameon": False, "lines.linewidth": 1.6})


def _save(fig, name: str) -> str:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=130)
    plt.close(fig)
    return f"realism/{name}"


def _local(ts: pd.Series, cfg: Config) -> pd.Series:
    return ts.dt.tz_convert(None) + pd.Timedelta(hours=cfg.site["location"]["utc_offset_h"])


# ------------------------------------------------------------------------------------------ figures
def fig_wind(truth: pd.DataFrame, cfg: Config) -> tuple[str, dict]:
    w = truth.set_index("ts")["wind_true"].resample("10min").mean().dropna().to_numpy()
    k_fit, lam_fit = weibull_fit(w)
    k_cfg, lam_cfg = cfg.site["wind"]["weibull_k"], cfg.site["wind"]["weibull_scale_ms"]
    x = np.linspace(0, max(w.max(), 12), 300)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.hist(w, bins=40, density=True, color=S1, alpha=0.35, edgecolor=SURFACE, linewidth=0.8, label="simulated 10-min means")
    ax.plot(x, stats.weibull_min.pdf(x, k_fit, scale=lam_fit), color=S1, label=f"fitted Weibull k={k_fit:.2f}, λ={lam_fit:.2f}")
    ax.plot(x, stats.weibull_min.pdf(x, k_cfg, scale=lam_cfg), color=S2, linestyle="--",
            label=f"configured hourly Weibull k={k_cfg}, λ={lam_cfg}")
    ax.set_xlabel("hub wind speed [m/s]")
    ax.set_ylabel("probability density")
    ax.set_title("Rooftop wind distribution (hub height)")
    ax.legend(fontsize=8)
    return _save(fig, "wind_weibull.png"), {"weibull_k_fit": k_fit, "weibull_scale_fit": lam_fit,
                                            "mean_wind": float(w.mean()), "p_above_3ms": float((w > 3).mean())}


def fig_diurnal(truth: pd.DataFrame, cfg: Config) -> str:
    df = truth[["ts", "wind_true", "ghi", "t_amb"]].copy()
    df["hour"] = _local(df["ts"], cfg).dt.hour
    g = df.groupby("hour")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, col, lab, c in zip(axes, ("wind_true", "ghi", "t_amb"),
                               ("hub wind [m/s]", "GHI [W/m²]", "ambient temperature [°C]"), (S1, S2, S3)):
        q = g[col].quantile([0.25, 0.5, 0.75]).unstack()
        ax.fill_between(q.index, q[0.25], q[0.75], color=c, alpha=0.18, linewidth=0, label="IQR")
        ax.plot(q.index, q[0.5], color=c, label="median")
        ax.set_xlabel("local hour")
        ax.set_title(lab)
        ax.set_xticks(range(0, 25, 6))
    axes[0].legend(fontsize=8)
    fig.suptitle("Diurnal cycles: calm mornings, afternoon sea breeze, summer heat", fontsize=10, color=INK2)
    return _save(fig, "diurnal.png")


def model_power_curve(cfg: Config, winds: np.ndarray) -> np.ndarray:
    """Steady-state electrical output [W] of the model at constant wind (no turbulence, 25 °C, ρ=1.225)."""
    t = cfg.turbine
    prm = K.build_params(t)
    st, vt = K.ocv_tables(t)
    prm[K.P_OVERSPEED_RPM] = 1e9  # show the aerodynamic curve beyond the trip
    out_p = []
    for v in winds:
        n = 240
        inp = K.default_inputs(n)
        state = K.initial_state(t, 25.0)
        state[K.S_OMEGA] = 5.0
        out = np.zeros((n, K.N_OUTPUTS))
        K.run_kernel(prm, st, vt, np.full(n * int(prm[K.P_SUBSTEPS]), v), inp, state, out,
                     np.zeros(n * int(prm[K.P_SUBSTEPS])), np.zeros(n * int(prm[K.P_SUBSTEPS])))
        out_p.append(out[-60:, K.O_P_WIND_OUT].mean())
    return np.asarray(out_p)


def fig_power_curve(tel: pd.DataFrame, truth: pd.DataFrame, cfg: Config) -> str:
    m = tel.drop_duplicates("seq").set_index(pd.to_datetime(tel.drop_duplicates("seq")["rx_time"]))
    one = m[["wind_ms", "power_w"]].resample("1min").mean().dropna()
    tr = truth.set_index("ts")[["wind_true", "p_wind_out"]].resample("1min").mean().dropna()
    v = np.linspace(0, 11, 45)
    pm = model_power_curve(cfg, v)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    axes[0].scatter(tr["wind_true"], tr["p_wind_out"], s=3, color=S1, alpha=0.15, linewidths=0, label="1-min means (truth)")
    axes[1].scatter(one["wind_ms"], one["power_w"], s=3, color=S2, alpha=0.15, linewidths=0, label="1-min means (as measured)")
    for ax in axes:
        ax.plot(v, pm, color=INK, linewidth=1.4, label="steady-state model")
        ax.axvline(cfg.turbine["rotor"]["cut_in_target_ms"], color=INK2, linestyle=":", linewidth=1)
        ax.set_xlabel("wind speed [m/s]")
        ax.set_xlim(0, 11)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("wind electrical power to bus [W]")
    axes[0].set_title("True hub wind vs output")
    axes[1].set_title("Cup anemometer vs INA219 (what the Pi sees)")
    return _save(fig, "power_curve.png")


def fig_temperatures(tel: pd.DataFrame, truth: pd.DataFrame, cfg: Config, days: float = 3.0) -> str:
    t0 = truth["ts"].iloc[0]
    tr = truth[truth["ts"] < t0 + pd.Timedelta(days=days)]
    te = tel.drop_duplicates("seq")
    te = te[pd.to_datetime(te["rx_time"], utc=True) < t0 + pd.Timedelta(days=days)]
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    x = _local(tr["ts"], cfg)
    axes[0].plot(x, tr["t_amb"], color=S3, label="ambient (truth)")
    axes[0].plot(x, tr["t_bearing"], color=S1, label="bearing (truth)")
    axes[0].plot(x, tr["t_gen"], color=S2, label="generator (truth)")
    axes[0].set_ylabel("°C")
    axes[0].set_title("Thermal model: solar heating by day, wind cooling, generator I²R")
    axes[0].legend(fontsize=8, ncol=3)
    xe = _local(pd.to_datetime(te["rx_time"], utc=True), cfg)
    ok = ~te["t_bearing_c"].isin(DS_GLITCH)
    axes[1].plot(xe[ok], te["t_bearing_c"][ok] - np.interp(te["seq"][ok], truth["seq"], truth["t_bearing"]),
                 color=S1, linewidth=0.8, label="DS18B20 bearing − truth (lag, offset, 0.0625 °C steps)")
    axes[1].set_ylabel("measurement error [°C]")
    axes[1].legend(fontsize=8)
    return _save(fig, "temperatures.png")


def fig_energy(truth: pd.DataFrame, cfg: Config) -> tuple[str, pd.DataFrame]:
    d = truth.set_index(_local(truth["ts"], cfg))[["p_wind_out", "p_pv", "p_load"]].resample("1D").sum() / 3600
    soc = truth.set_index(_local(truth["ts"], cfg))["soc"].resample("10min").mean()
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    wbar = 0.28
    for i, (col, c, lab) in enumerate((("p_wind_out", S1, "wind"), ("p_pv", S2, "PV"), ("p_load", S3, "self-consumption"))):
        axes[0].bar(d.index + pd.Timedelta(hours=(i - 1) * 7), d[col], width=wbar, color=c, label=lab)
    axes[0].set_ylabel("energy per day [Wh]")
    axes[0].set_title("Daily energy: PV dominates on the rooftop; the Pi-class load is significant")
    axes[0].legend(fontsize=8, ncol=3)
    axes[1].plot(soc.index, soc * 100, color=S4)
    axes[1].set_ylabel("battery SOC [%]")
    axes[1].set_ylim(0, 100)
    return _save(fig, "energy.png"), d


# ------------------------------------------------------------------------------------------ tables
def sensor_table(tel: pd.DataFrame, truth: pd.DataFrame, cfg: Config) -> list[tuple[str, str, str]]:
    n_true = len(truth)
    uniq = tel.drop_duplicates("seq")
    dup = len(tel) - len(uniq)
    seq_arrival = tel["seq"].to_numpy()
    late = int((np.diff(seq_arrival) < 0).sum())
    ds_cols = ("t_gen_c", "t_bearing_c", "t_amb_c")
    n85 = int(sum((uniq[c] == 85.0).sum() for c in ds_cols))
    n127 = int(sum((uniq[c] == -127.0).sum() for c in ds_cols))
    sat = float((uniq["lux"] >= cfg.sensors["bh1750"]["saturation_lux"]).mean())
    day = uniq["lux"] > 1000
    sat_day = float((uniq.loc[day, "lux"] >= cfg.sensors["bh1750"]["saturation_lux"]).mean()) if day.any() else 0.0
    j = uniq.merge(truth[["seq", "wind_true", "rpm"]], on="seq")
    w_bias = float((j["wind_ms"] - j["wind_true"]).mean())
    moving = j["wind_true"] > 3
    w_bias_m = float((j.loc[moving, "wind_ms"] - j.loc[moving, "wind_true"]).mean())
    rpm_err = float(np.abs(j["rpm_x"] - j["rpm_y"]).mean())   # _x measured, _y truth
    dht_stale = float((uniq["dht_t_c"].diff() == 0).mean())
    return [
        ("Packets lost (Wi-Fi, burst model)", f"{100 * (1 - len(uniq) / n_true):.2f} %", "Gilbert-Elliott: losses arrive in bursts"),
        ("Duplicate packets (QoS 1)", f"{dup}", "same seq delivered twice"),
        ("Out-of-order arrivals", f"{late}", "late packets overtaken by newer ones"),
        ("DS18B20 85.0 °C power-on reads", f"{n85}", "across 3 probes"),
        ("DS18B20 −127 °C disconnect reads", f"{n127}", "incl. whole-bus dropouts"),
        ("BH1750 saturated (all / daytime)", f"{100 * sat:.1f} % / {100 * sat_day:.1f} %", "65535 lx ≈ 546 W/m² at /120"),
        ("Anemometer bias vs hub truth (all / v>3)", f"{w_bias:+.2f} / {w_bias_m:+.2f} m/s", "overspeeding + quantisation + stall"),
        ("Hall rpm |error| (count method)", f"{rpm_err:.1f} rpm", "±60 rpm steps at 1 pulse/rev, 1 s gate"),
        ("DHT22 repeated values", f"{100 * dht_stale:.0f} %", "2 s min interval + failed reads"),
    ]


# ------------------------------------------------------------------------------------------ html
def write_html(sections: list[tuple[str, str, str]], tables: dict[str, list[tuple[str, ...]]], meta: dict) -> Path:
    css = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--rule:#e4e3df;--accent:#2a78d6}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--rule:#33332f}}
:root[data-theme="dark"]{--bg:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--rule:#33332f}
body{background:var(--bg);color:var(--ink);font:15px/1.55 Inter,system-ui,sans-serif;margin:0;padding:0 16px}
main{max-width:1100px;margin:32px auto}h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 6px}
p,li{color:var(--ink2)}img{max-width:100%;border:1px solid var(--rule);border-radius:6px;background:#fcfcfb}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid var(--rule);padding:6px 8px;text-align:left}
td.num{font-family:'JetBrains Mono',ui-monospace,monospace;font-variant-numeric:tabular-nums;white-space:nowrap}
.meta{font-family:ui-monospace,monospace;font-size:12px;color:var(--ink2)}"""
    parts = [f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
             f"<title>Simulator Realism Check</title><style>{css}</style></head><body><main>",
             "<h1>AeroSolar Twin: simulator realism check</h1>",
             f"<p class=meta>{html.escape(json.dumps(meta))}</p>",
             "<p>Everything below is <b>simulated</b>. The purpose is to show that the synthetic data has the "
             "statistical and physical character of a real rooftop VAWT, so models trained on it transfer "
             "to the prototype after recalibration (see CALIBRATION.md).</p>"]
    for title, img, text in sections:
        parts.append(f"<h2>{html.escape(title)}</h2><p>{text}</p><img src='{img}' alt='{html.escape(title)}'>")
    for title, rows in tables.items():
        parts.append(f"<h2>{html.escape(title)}</h2><table>")
        for r in rows:
            parts.append("<tr><td>" + html.escape(r[0]) + "</td><td class=num>" + html.escape(r[1]) + "</td><td>"
                         + html.escape(r[2] if len(r) > 2 else "") + "</td></tr>")
        parts.append("</table>")
    parts.append("</main></body></html>")
    path = REPORT_DIR / "realism.html"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=REPO_ROOT / "data" / "fc_2026_06_s206")
    a = ap.parse_args(argv)
    _style()
    cfg = Config.load()
    if not (a.run / "truth_1hz.parquet").exists():
        print(f"{a.run.name} has no truth_1hz.parquet (deleted to save disk space). Regenerate this run with:")
        print(f"  python -m aerosolar.sim.generate --preset month --start <first day of the month> --seed <seed>")
        return
    truth = pd.read_parquet(a.run / "truth_1hz.parquet")
    tel = pd.read_parquet(a.run / "telemetry_1hz.parquet")
    img_w, wstats = fig_wind(truth, cfg)
    img_d = fig_diurnal(truth, cfg)
    img_p = fig_power_curve(tel, truth, cfg)
    img_t = fig_temperatures(tel, truth, cfg)
    img_e, daily = fig_energy(truth, cfg)
    brake_frac = float(truth["brake"].mean())
    sections = [
        ("Wind distribution", img_w,
         f"10-minute hub-height means over the run, mean {wstats['mean_wind']:.2f} m/s, "
         f"{100 * wstats['p_above_3ms']:.0f} % of the time above 3 m/s. The fitted Weibull differs from the "
         "configured hourly one because diurnal and shamal multipliers reshape it; turbulence widens the tail."),
        ("Diurnal cycles", img_d, "Median and interquartile range by local hour. Afternoon sea breeze peaks "
         "around 15:00; mornings are calm; irradiance follows solar geometry for 25.1° N with dust/cloud dips."),
        ("Power curve", img_p, "Left: truth. Right: what the Pi sees (cup anemometer overspeeding and 0.67 m/s "
         "quantisation smear the curve). The electrical cut-in (~4.5–5 m/s) is far above the rotational cut-in "
         "because the PMG only charges once its rectified EMF exceeds the battery voltage. Output stops "
         f"above ~8 m/s because the 200 rpm overspeed trip engages (braked {100 * brake_frac:.2f} % of this run)."),
        ("Temperatures", img_t, "Bearing and generator temperatures ride on ambient plus solar heating of the mast, "
         "cooled by wind. The lower panel shows DS18B20 error: probe lag, per-device offset and 0.0625 °C steps "
         "(85/−127 glitch codes are removed here and counted in the table)."),
        ("Energy balance", img_e, "Daily energy from the truth record. Rooftop wind contributes little at a 4.5 m/s "
         "Weibull scale; the 12 V self-consumption (Pi 5 + ESP32 + Arduino + sensors) is comparable to the PV yield."),
    ]
    tables = {"Sensor and network quirks": sensor_table(tel, truth, cfg),
              "Daily energy [Wh] (first 7 days)": [(str(i.date()), f"wind {r.p_wind_out:6.1f}  PV {r.p_pv:6.1f}  load {r.p_load:6.1f}", "")
                                                   for i, r in daily.head(7).iterrows()]}
    meta = json.loads((a.run / "meta.json").read_text())
    path = write_html(sections, tables, {"run": meta["preset"], "seed": meta["seed"], "days": meta["duration_h"] / 24})
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
