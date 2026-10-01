"""Forecasting data campaign: two simulated years of month-long runs.

* **Year A (2025)** -- training. The last 5 days of every month are held out for model selection and for
  calibrating the prediction intervals (``CALIB_LAST_DAYS``).
* **Year B (2026)** -- test. Never used for fitting, tuning or calibration.

Every month is a separate run with its own seed (independent weather realisation), so year B is not a
replay of year A. Splitting by *year* keeps the evaluation honest: the test months are later in time and
cover every season.

    python -m aerosolar.ml.campaign --workers 7
"""
from __future__ import annotations

import argparse
import calendar
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ..config import REPO_ROOT

DATA_DIR = REPO_ROOT / "data"
YEARS = {"train": 2025, "test": 2026}
SEED_BASE = {"train": 100, "test": 200}
CALIB_LAST_DAYS = 5


def runs() -> list[dict]:
    out = []
    for split, year in YEARS.items():
        for month in range(1, 13):
            days = calendar.monthrange(year, month)[1]
            out.append({"split": split, "year": year, "month": month, "seed": SEED_BASE[split] + month,
                        "start": f"{year}-{month:02d}-01T00:00:00", "hours": days * 24,
                        "name": f"fc_{year}_{month:02d}"})
    return out


def run_dir(r: dict) -> Path:
    return DATA_DIR / f"{r['name']}_s{r['seed']}"


def _job(r: dict) -> str:
    from ..sim.generate import run
    return str(run("month", seed=r["seed"], hours=r["hours"], start=r["start"], name=r["name"], quiet=True))


def generate_missing(workers: int = 6, force: bool = False) -> None:
    todo = [r for r in runs() if force or not (run_dir(r) / "meta.json").exists()]
    print(f"{len(todo)} month runs to generate with {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for path in ex.map(_job, todo):
            print("  wrote", path, flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    generate_missing(a.workers, a.force)


if __name__ == "__main__":
    main()
