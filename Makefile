PY ?= python

.PHONY: install test lint demo app synthetic

install:
	$(PY) -m pip install -e ".[all]"

test:
	$(PY) -m pytest -q

lint:
	ruff check .

# auto-label the bundled example picks: real conveyor frames + synthetic tray (CPU, ~10 s)
demo:
	picklabel autolabel examples --out runs/autolabel/examples

app:
	streamlit run app/streamlit_app.py

# the full loop on synthetic data: auto-label -> datasets -> two students -> eval (~1 h on an M1)
synthetic:
	$(PY) scripts/synthetic_e2e.py --work runs/synthetic_e2e --report report/synthetic
