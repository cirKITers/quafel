# QUAFEL - QUAntum Framework EvaLuation

Not to be confused with the [Quaffle](https://harrypotter.fandom.com/wiki/Quaffle) ;)

*This project follows the [Kedro Framework](https://kedro.org/) approach.*

## Setup

Straight, **without development packages**, you can execute the following command, assuming you have [Poetry](https://python-poetry.org/) installed:
```
poetry install --without dev
```
There is a ```setup.sh``` script in the ```.vscode``` directory for convenience.

If you considere building docs, running tests and commiting to the project, run:
```
poetry install
poetry run pre-commit autoupdate
poetry run pre-commit install
poetry run pytest
poetry run mkdocs build
```
Again, there is a ```setup_dev.sh``` script in the ```.vscode``` directory for convenience.

## Usage

Without any configuration needed, you can execute
```
poetry kedro run
```
(or omit poetry if you're using classical venvs) and a default pipeline should run.
By now the default pipeline is the only one available.

Checkout the pre-defined VSCode tasks if you want to develop on the project.

### Tuning the test circuits

Circuits are being generated in the ```data_generation``` namespace of the project.
To adjust the number of qubits, depth of the circuit and other parameters, checkout [conf/base/parameters/data_generation.yml](/conf/base/parameters/data_generation.yml).

### Selecting a Framework and Execution behaviour

Everything related to executing the circuits and time measurments is contained in the ```data_science``` namespace.
Head to [conf/base/parameters/data_science.yml](/conf/base/parameters/data_science.yml) to specify a framework and set e.g. the number of evaluations.