"""Build, execute and export the explanatory notebooks.

    python -m aerosolar.tools.run_notebooks            # all
    python -m aerosolar.tools.run_notebooks 03 05      # selected

Writes ``notebooks/NN_name.ipynb`` (executed, outputs saved) and ``reports/notebooks/NN_name.html``.
The notebook sources live in ``nb_sources.py`` so they are reviewable as plain Python and never hold logic.
"""
from __future__ import annotations

import sys
import time

import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor

from ..config import REPO_ROOT
from . import nb_sources as N

NOTEBOOKS = {
    "01_turbine_power_curve_and_sensors": N.NB01,
    "02_simulated_data_and_dataset": N.NB02,
    "03_day_ahead_forecast": N.NB03,
    "04_nowcast": N.NB04,
    "05_retrain_on_real_data": N.NB05,
}
NB_DIR = REPO_ROOT / "notebooks"
HTML_DIR = REPO_ROOT / "reports" / "notebooks"
TIMEOUT_S = 1800


def build(name: str, cells: list[tuple[str, str]]) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "aerosolar", "display_name": "AeroSolar (.venv)", "language": "python"}
    nb.cells = [nbformat.v4.new_markdown_cell(t) if k == "md" else nbformat.v4.new_code_cell(t) for k, t in cells]
    return nb


def run(selected: list[str] | None = None) -> None:
    NB_DIR.mkdir(exist_ok=True)
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        if selected and not any(name.startswith(s) for s in selected):
            continue
        t0 = time.time()
        nb = build(name, cells)
        ExecutePreprocessor(timeout=TIMEOUT_S, kernel_name="aerosolar").preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
        nbformat.write(nb, NB_DIR / f"{name}.ipynb")
        body, _ = HTMLExporter().from_notebook_node(nb)
        (HTML_DIR / f"{name}.html").write_text(body, encoding="utf-8")
        print(f"{name}: {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    run(sys.argv[1:] or None)
