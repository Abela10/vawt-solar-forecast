"""Batch dataset generation from a scenario.

Usage::

    python -m aerosolar.sim.generate --preset summer_30d --seed 1
    python -m aerosolar.sim.generate --preset month --start 2026-03-01T00:00:00 --seed 5

Outputs in ``data/<run>/``:

* ``telemetry_1hz.parquet``        ESP32 packets as received by the Pi (+ ``rx_time``), arrival order
* ``status_1hz.parquet``           Pi-side controller status (brake, charge state, load shedding)
* ``truth_1hz.parquet``            physical ground truth (evaluation only -- never a model input)
* ``forecast_issued.parquet``      hourly-issued synthetic day-ahead weather forecasts (Open-Meteo schema)
* ``weather_hourly_truth.parquet`` what the weather actually did, hourly
* ``meta.json``                    scenario, seed, energy summary, config snapshot
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .. import __version__
from ..config import REPO_ROOT, Config
from .engine import Engine
from .forecasts import issue_forecasts

CHUNK_S = 3600
DATA_DIR = REPO_ROOT / "data"
FLOAT_DOWNCAST_EXCLUDE = ("t",)


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    """float64 -> float32 for storage (the time column keeps full precision)."""
    for c in df.columns:
        if df[c].dtype == np.float64 and c not in FLOAT_DOWNCAST_EXCLUDE:
            df[c] = df[c].astype(np.float32)
    return df


class _Writer:
    """Append-only Parquet writer (streams chunks so long runs fit in memory)."""

    def __init__(self, path: Path):
        self.path = path
        self.w: pq.ParquetWriter | None = None

    def write(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self.w is None:
            self.w = pq.ParquetWriter(self.path, table.schema, compression="zstd")
        else:
            table = table.cast(self.w.schema)
        self.w.write_table(table)

    def close(self) -> None:
        if self.w is not None:
            self.w.close()


def run(preset: str | dict, seed: int = 0, out: Path | None = None, hours: float | None = None, quiet: bool = False,
        cfg: Config | None = None, start: str | None = None, name: str | None = None) -> Path:
    """Generate a dataset for ``preset`` (a name in scenarios.yaml, or a preset dict) and return its folder."""
    cfg = cfg or Config.load()
    eng = Engine(cfg, preset, seed=seed, start=start)
    name = name or (preset if isinstance(preset, str) else preset.get("name", "custom"))
    dur_h = float(hours if hours is not None else eng.duration_h or 24)
    out = Path(out) if out else DATA_DIR / f"{name}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    writers = {k: _Writer(out / f"{k}.parquet") for k in ("telemetry_1hz", "status_1hz", "truth_1hz")}
    total_s = int(round(dur_h * 3600))
    t_wall = time.time()
    energy = {"wind_wh": 0.0, "pv_wh": 0.0, "load_wh": 0.0, "braked_s": 0}
    done = 0
    while done < total_s:
        n = min(CHUNK_S, total_s - done)
        r = eng.step(n)
        writers["telemetry_1hz"].write(_downcast(r.packets))
        writers["status_1hz"].write(r.status)
        writers["truth_1hz"].write(_downcast(r.truth))
        energy["wind_wh"] += float(r.truth["p_wind_out"].sum()) / 3600
        energy["pv_wh"] += float(r.truth["p_pv"].sum()) / 3600
        energy["load_wh"] += float(r.truth["p_load"].sum()) / 3600
        energy["braked_s"] += int(r.truth["brake"].sum())
        done += n
        if not quiet:
            sys.stdout.write(f"\r[{name}] {done / 3600:7.1f}/{dur_h:.0f} h  {time.time() - t_wall:6.1f} s")
            sys.stdout.flush()
    for w in writers.values():
        w.close()
    if not quiet:
        print()
    fc, ht = issue_forecasts(eng.weather, int(math.ceil(dur_h)), seed=seed)
    fc.to_parquet(out / "forecast_issued.parquet", index=False)
    ht.to_parquet(out / "weather_hourly_truth.parquet", index=False)
    meta = {"preset": name, "preset_def": preset if isinstance(preset, dict) else None, "seed": seed,
            "version": __version__, "start_local": eng.start_local.isoformat(), "start_utc": str(eng.start_utc),
            "duration_h": dur_h, "wall_time_s": time.time() - t_wall, "energy": energy, "config": eng.cfg.to_dict()}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hours", type=float, default=None, help="override scenario duration")
    ap.add_argument("--start", default=None, help="local start time, ISO format")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    print(f"wrote {run(a.preset, a.seed, a.out, a.hours, start=a.start)}")


if __name__ == "__main__":
    main()
