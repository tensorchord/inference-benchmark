PY_SOURCE=src benchmark

lint:
	@black --check --diff ${PY_SOURCE}
	@ruff check ${PY_SOURCE}

format:
	@black ${PY_SOURCE}
	@ruff check --fix ${PY_SOURCE}

clean:
	@-rm -rf dist build __pycache__ src/*.egg-info src/_version.py

build:
	@python -m build
