.PHONY: black-check
black-check:
	poetry run black --check .

.PHONY: black
black:
	poetry run black .

.PHONY: ruff
ruff:
	poetry run ruff check . --fix

.PHONY: ruff-check
ruff-check:
	poetry run ruff check .

.PHONY: mdformat
mdformat:
	poetry run mdformat *.md

.PHONY: mdformat-check
mdformat-check:
	poetry run mdformat --check *.md

.PHONY: mypy
mypy:
	poetry run mypy .

.PHONY: cake
cake:
	$(MAKE) black
	$(MAKE) ruff
	$(MAKE) mdformat

.PHONY: lint
lint:
	$(MAKE) black-check
	$(MAKE) ruff-check
	$(MAKE) mdformat-check
	$(MAKE) mypy