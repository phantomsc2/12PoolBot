.PHONY: fix check

POETRY ?= poetry
PYTHON = $(POETRY) run python
SOURCES = bot scripts

fix:
	$(PYTHON) -m ruff format $(SOURCES)
	$(PYTHON) -m ruff check --fix --unsafe-fixes $(SOURCES)

check:
	$(PYTHON) -m ruff check $(SOURCES)
	$(PYTHON) -m ruff format --check $(SOURCES)
	$(PYTHON) -m mypy $(SOURCES)
