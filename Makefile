.PHONY: setup check test

setup:
	python -m pip install -r requirements.txt

check:
	python scripts/check_environment.py

test:
	pytest -q
