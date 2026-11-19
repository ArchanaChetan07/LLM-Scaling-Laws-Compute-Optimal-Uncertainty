PY ?= python
export PYTHONPATH := src
export PYTHONIOENCODING := utf-8

.PHONY: data build test quick all check verify figures report
data:        ## download raw inputs (~600 MB)
	bash scripts/download_data.sh
build:       ## raw -> data/processed
	$(PY) scripts/build_dataset.py
test:
	$(PY) -m pytest -q tests
quick:       ## ~10 min smoke run (B=60)
	$(PY) scripts/run_all.py --quick
all: build   ## headline + OAT + corpora (B=1000 / 300), factorial, coverage, figures
	$(PY) scripts/validate_schedule.py
	$(PY) scripts/run_all.py
	$(PY) scripts/run_factorial.py
	$(PY) scripts/coverage_sim.py
	$(PY) scripts/derive.py
	$(PY) -m scalingci.plots
	$(PY) scripts/render_readme.py
check:       ## README numbers == results/
	$(PY) scripts/render_readme.py --check
verify:      ## GPU: independently re-measure Pythia losses on Pile val (~3 GB of checkpoints)
	$(PY) scripts/verify_pythia_losses.py
figures:
	$(PY) -m scalingci.plots
report:      ## README + interactive HTML report from results/
	$(PY) scripts/derive.py
	$(PY) scripts/render_readme.py
	$(PY) scripts/build_web_report.py
