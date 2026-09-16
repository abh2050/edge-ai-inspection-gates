SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help
.NOTPARALLEL:
UV ?= uv
RUN = $(UV) run --frozen

.PHONY: help lock install fetch capture-ane capture-ane-neuralnetwork gate0 gate1 gate2 gate3 gate4 gate5 gates test lint agent-plan agent-run release-check dashboard dashboard-build
help:
	@echo 'Run make lock once, then make install. Implement the contracts before running gates.'
	@echo 'Run make fetch after recording an approved URL and SHA256 in data/manifest.json.'
	@echo 'Run make gates on the actual Apple Silicon measurement host.'
lock:
	$(UV) lock
install:
	$(UV) sync --frozen --extra llm
fetch:
	$(RUN) cycletime fetch
capture-ane:
	sudo .venv/bin/python scripts/capture_ane_evidence.py
capture-ane-neuralnetwork:
	sudo .venv/bin/python scripts/capture_ane_evidence.py --model-format NeuralNetwork
# Each gate validates its own prior evidence without repeating earlier computation.
gate0:
	$(RUN) cycletime gate 0
gate1:
	$(RUN) cycletime gate 1
gate2:
	$(RUN) cycletime gate 2
gate3:
	$(RUN) cycletime gate 3
gate4:
	$(RUN) cycletime gate 4
gate5:
	$(RUN) cycletime gate 5
gates:
	$(MAKE) gate0
	$(MAKE) gate1
	$(MAKE) gate2
	$(MAKE) gate3
	$(MAKE) gate4
	$(MAKE) gate5
test:
	$(RUN) pytest
lint:
	$(RUN) ruff check .
agent-plan:
	$(RUN) cycletime agent plan
agent-run:
	$(RUN) cycletime agent run
release-check:
	$(RUN) cycletime release-check
# The dashboard renders recorded evidence on the local host only. ADR 0008 bounds its scope.
dashboard-build:
	$(RUN) python scripts/build_dashboard.py
dashboard: dashboard-build
	@echo 'Serving the evidence dashboard at http://127.0.0.1:8787/ (ctrl-c to stop).'
	@cd dashboard && $(UV) run --frozen python -m http.server 8787 --bind 127.0.0.1
