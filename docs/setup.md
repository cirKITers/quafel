# Setup

This project follows the [Kedro Framework](https://kedro.org/) approach.
Straight, **without development packages**, you can execute the following command, assuming you have [Poetry](https://python-poetry.org/) installed:
```
poetry install --without dev
```
There is a ```setup.sh``` script in the ```.vscode``` directory for convenience.

If you want to go with Pip instead, run 
```
pip install -r src/requirements.in
```