# QUAFEL - QUAntum Framework EvaLuation

Not to be confused with the [Quaffle](https://harrypotter.fandom.com/wiki/Quaffle) :wink:

![Overview](docs/overview.png)

*This project follows the [Kedro Framework](https://kedro.org/) approach.*

## :hammer: Setup

Straight, **without development packages**, you can execute the following command, assuming you have [Poetry](https://python-poetry.org/) installed:
```
poetry install --without dev
```
There is a ```setup.sh``` script in the ```.vscode``` directory for convenience.

***

:construction: only:

If you considere building docs, running tests and commiting to the project, run:
```
poetry install
poetry run pre-commit autoupdate
poetry run pre-commit install
poetry run pytest
poetry run mkdocs build
```
Again, there is a ```setup_dev.sh``` script in the ```.vscode``` directory for convenience.

***

## :rocket: Usage

Without any configuration needed, you can execute
```
poetry kedro run
```
(or omit poetry if you're using classical venvs) and a default pipeline should run.

This project can take advantage of multiprocessing to evaluate numerous combinations of *qubits*, *depths* and *shots*.
To enable this, you must run
```
poetry kedro run --pipeline pre
```
which will generate a [Partitioned Dataset]() from which a parallel runner can spawn individual processes for each configuration defined by the above mentioned parameters.
This dataset must be re-generated after tuning those parameters.
After doing so, you can run
```
poetry kedro run --pipeline parallel --runner ParallelRunner
```
which will calculate the duration and result for each configuration.
For details on the output, see the [Data Structure Section](#floppy_disk-data-structure).
As the files are just named by ids, you might want to execute
```
poetry kedro run --pipeline visualize
```
to view those evaluation results.

***
:construction: only:

Checkout the pre-defined VSCode tasks if you want to develop on the project.

***
### Tuning the test circuits

Circuits are being generated in the ```data_generation``` namespace of the project.
To adjust the number of qubits, depth of the circuit and other parameters, checkout [conf/base/parameters/data_generation.yml](/conf/base/parameters/data_generation.yml).

### Selecting a Framework and Execution behaviour

Everything related to executing the circuits and time measurments is contained in the ```data_science``` namespace.
Head to [conf/base/parameters/data_science.yml](/conf/base/parameters/data_science.yml) to specify a framework and set e.g. the number of evaluations.

### :eyeglasses: Pipeline

You can actually see what's going on by running
```
poetry run kedro-viz
```
which will open a browser with [kedro-viz](https://github.com/kedro-org/kedro-viz) showing the pipeline.

![kedro-viz view of the pipeline](docs/kedro_view.png)

### :floppy_disk: Data Structure

- [data/01_raw](data/01_raw):
  - [Evaluation Matrix](data/01_raw/dataset.json) containing all valid values for ```frameworks```, ```qubits```, ```depths```, and ```shots``` as specified in the [data_generation.yml](conf/base/parameters/data_generation.yml) file.
- [data/02_intermediate](data/02_intermediate):
  - Evaluation Partitions split into single ```.csv``` files
- [data/03_qasm_circuits](data/03_qasm_circuits/):
  - as the name suggests, all generated qasm circuits for the job with the corresponding id
- [data/04_execution_results](data/04_execution_results/):
  - simulator results from the job with the corresponding id.
- [data/05_execution_durations](data/05_execution_durations/):
  - duration results from the job with the corresponding id.
- [data/06_executions_combined](data/06_executions_combined/):
  - **Versioned** dataset containing the combined information of both, the input parameters (```framework```, ```qubits```, ```depth```, ```shots```) and the measured duration
- [data/07_reportings](data/07_reporting):
  - **Versioned** dataset with the ```.json``` formatted ploty heatmaps
  - The data in this folder is named by the framework and the fixed parameter. E.g. when the number of ```qubits``` is plotted against the ```shots``` and the ```qiskit_fw``` is being used to simulate a circuit of ```depth``` $3$, the filename would be ```qiskit_fw_depth_3```.
