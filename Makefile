PY ?= .venv/Scripts/python

.PHONY: test data train evaluate notebooks realism all

test:
	$(PY) -m pytest

data:
	$(PY) -m aerosolar.ml.campaign --workers 7

train:
	$(PY) -m aerosolar.ml.train --workers 7

evaluate:
	$(PY) -m aerosolar.ml.evaluate

notebooks:
	$(PY) -m aerosolar.tools.run_notebooks

realism:
	$(PY) -m aerosolar.sim.realism

all: data train evaluate notebooks realism
