# Contributing to QUAFEL

:tada: Welcome! :tada:

Contributions are highly welcome!
Start of by..
1. Creating an issue using one of the templates (Bug Report, Feature Request)
   - let's discuss what's going wrong or what should be added
   - can you contribute with code? Great! Go ahead! :rocket:
2. Forking the repository and working on your stuff
3. Creating a pull request to the main repository

## Setup

In addition to what is mentioned in the [`README.md`]() 

If you considere building docs, running tests and commiting to the project, run:
```
poetry install
poetry run pre-commit autoupdate
poetry run pre-commit install
poetry run pytest
poetry run mkdocs build
```
Again, there is a ```setup_dev.sh``` script in the ```.vscode``` directory for convenience.

With Pip the equivalent is
```
pip install -r src/requirements_dev.in
pre-commit autoupdate
pre-commit install
pytest
mkdocs build
```

## Developing

Checkout the debugging configurations in `.vscode` folder (if you're using that IDE).