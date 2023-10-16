poetry install
poetry run pre-commit autoupdate
poetry run pre-commit install
poetry run pytest
poetry run mkdocs build