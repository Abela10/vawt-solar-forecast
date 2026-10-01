"""Reproducible training of the power-forecast models.

    python -m aerosolar.ml.train [--seed 0] [--workers 7]

    month runs ─► clean 1 Hz ─► hourly actuals ─┐
    issued forecasts ───────────────────────────┴► feature table ─► DayAheadForecaster (PV, wind, total)
    month runs ─► clean 1-minute table ─► nowcast table ─► Nowcaster (5 min)

Split (see :mod:`aerosolar.ml.campaign`): year A minus its last 5 days per month = train; those days =
calibration (early stopping + conformal intervals); year B = test, untouched here.
Artifacts: ``models/forecast_<version>.joblib`` + ``.json`` metadata; per-run tables cached in
``data/<run>/derived/``.
"""
from __future__ import annotations

import argparse
import calendar
import json
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .. import __version__
from ..config import REPO_ROOT, Config
from . import campaign as C
from .data_quality import clean
from .dataset import load_run_1hz, minute_table
from .forecast_day import DayAheadForecaster
from .forecast_features import FEATURES, TABLE_VERSION, add_derived, run_table
from .nowcast import Nowcaster, nowcast_table

MODELS_DIR = REPO_ROOT / "models"
NOWCAST_TRAIN_STRIDE_MIN = 2          # every 2nd minute is enough for training (neighbours are near-duplicates)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _prepare(run_path: str) -> str:
    """Build and cache the feature table, hourly actuals and the nowcast table of one month run."""
    run = Path(run_path)
    d = run / "derived"
    d.mkdir(exist_ok=True)
    stamp = d / "VERSION"
    if (d / "nowcast.parquet").exists() and stamp.exists() and stamp.read_text().strip() == TABLE_VERSION:
        return run_path
    cfg = Config.load()
    table, hourly = run_table(run, cfg)
    table.to_parquet(d / "table.parquet", index=False)
    hourly.to_parquet(d / "hourly.parquet")
    df1, _ = clean(load_run_1hz(run))
    nowcast_table(minute_table(df1, cfg)).to_parquet(d / "nowcast.parquet")
    stamp.write_text(TABLE_VERSION)
    return run_path


def prepare_all(workers: int = 6) -> None:
    paths = [str(C.run_dir(r)) for r in C.runs() if (C.run_dir(r) / "meta.json").exists()]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for p in ex.map(_prepare, paths):
            _log(f"  prepared {Path(p).name}")


def is_calibration(valid_time: pd.Series, utc_offset_h: float) -> np.ndarray:
    """True for valid times in the last CALIB_LAST_DAYS local days of their month."""
    local = pd.DatetimeIndex(valid_time).tz_convert(None) + pd.Timedelta(hours=utc_offset_h)
    dim = np.array([calendar.monthrange(y, m)[1] for y, m in zip(local.year, local.month)])
    return local.day > dim - C.CALIB_LAST_DAYS


def load_split(split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenated (day-ahead table, nowcast table) of all runs of a split, tagged with the run name."""
    tabs, ncs = [], []
    for r in C.runs():
        if r["split"] != split:
            continue
        d = C.run_dir(r) / "derived"
        if not (d / "table.parquet").exists():
            continue
        t = add_derived(pd.read_parquet(d / "table.parquet"))
        if "y_float" not in t:
            hourly = pd.read_parquet(d / "hourly.parquet")
            t["y_float"] = hourly["float_frac"].reindex(pd.DatetimeIndex(t["valid_time"])).to_numpy()
        t["run"] = r["name"]
        tabs.append(t)            # rows past the run's end have NaN targets and are dropped by the models
        n = pd.read_parquet(d / "nowcast.parquet")
        n["run"] = r["name"]
        ncs.append(n)
    return pd.concat(tabs, ignore_index=True), pd.concat(ncs)


def train(seed: int = 0, workers: int = 6, prepare_only: bool = False) -> Path:
    cfg = Config.load()
    t0 = time.time()
    _log("preparing per-run tables")
    prepare_all(workers)
    if prepare_only:
        return MODELS_DIR
    table, nc = load_split("train")
    off = cfg.site["location"]["utc_offset_h"]
    cal = is_calibration(table["valid_time"], off)
    _log(f"day-ahead: {int((~cal).sum())} train rows, {int(cal.sum())} calibration rows")
    fda = DayAheadForecaster().fit(table[~cal], table[cal], seed=seed, log=_log)
    nc_cal = is_calibration(nc["ts"], off)
    nc_tr = nc[~nc_cal].iloc[::NOWCAST_TRAIN_STRIDE_MIN]
    _log(f"nowcast: {len(nc_tr)} train rows")
    nowcaster = Nowcaster(seed=seed).fit(nc_tr, nc[nc_cal])

    MODELS_DIR.mkdir(exist_ok=True)
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = MODELS_DIR / f"forecast_{version}.joblib"
    joblib.dump({"day_ahead": fda, "nowcaster": nowcaster}, path, compress=3)
    meta = {"version": version, "package_version": __version__, "seed": seed,
            "train_runs": sorted(table["run"].unique().tolist()), "calibration": f"last {C.CALIB_LAST_DAYS} days of each month",
            "test_year": C.YEARS["test"], "features_day_ahead": FEATURES, "features_nowcast": nowcaster.features,
            "trees_per_member": {t: [int(x.best_iteration_) for x in m.members] for t, m in fda.models.items()},
            "conformal_widening_w": {t: {f"{k[0]}%_block{k[1]}": v for k, v in m.conformal.items()} for t, m in fda.models.items()},
            "train_time_s": time.time() - t0}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    (MODELS_DIR / "LATEST").write_text(path.name, encoding="utf-8")
    _log(f"saved {path.name} in {time.time() - t0:.0f} s")
    return path


def load_latest() -> tuple[dict, dict]:
    name = (MODELS_DIR / "LATEST").read_text(encoding="utf-8").strip()
    path = MODELS_DIR / name
    return joblib.load(path), json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    train(a.seed, a.workers)


if __name__ == "__main__":
    main()
