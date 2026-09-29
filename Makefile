.PHONY: install data report test lint

install:
	pip install -e ".[data,dev]"

data:
	python scripts/download_data.py --start 20180101 --end 20250630

report:
	python -m qfp.pipeline --config configs/default.yaml

test:
	pytest -q

lint:
	ruff check qfp tests scripts
