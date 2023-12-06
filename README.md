# QUAFEL - QUAntum Framework EvaLuation

Not to be confused with the [Quaffle](https://harrypotter.fandom.com/wiki/Quaffle) :wink:

![Overview](docs/overview.png)

See [our poster](https://bwsyncandshare.kit.edu/s/CdnD6MEsNwYgJMd) contributed to the [QTML23](https://indico.cern.ch/event/1288979/) for more details.


## Setup :hammer:

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

<details>
<summary>:construction: only:</summary>

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
</details>

## Usage :rocket: 

Without any configuration needed, you can execute
```
kedro run --pipeline prepare
```
followed by
```
kedro run
```
and a default pipeline should run. In this and following examples the leading `poetry run` is omitted for simplicity.

Note that is required to always run the `prepare` pipeline in advance to any actual processing pipeline.
This is because of the current implementation relies on dynamically created nodes that are depending on the configuration and therefore requiring two separate pipeline executions.

In summary, the following pipelines exist:
- `prepare` : generates all possible combinations of configurations based on the current parameter set
- `measure` : performs the actual time measurement by executing experiments for each of the previously generated configurations with the ability to parallelize processing
- `ctmeasure` : if the `measure` pipeline failed or was cancelled, use this pipeline to resume
- `combine` : gathers all the results from the `measure` pipeline and combines them into a single output dataset
- `visualize` : takes the combined experiment results and generates your plots

The `default` pipeline covers `measure`, `combine` and `visualize`.
You can run them separately by specifying the pipeline name.

This project can take advantage of multiprocessing to evaluate numerous combinations of *qubits*, *depths* and *shots* in parallel in the `measure` pipeline.
To use this, you should explicitly call the individual pipelines.
In summary the whole experiment will then look as follows:
```
kedro run --pipeline prepare
kedro run --pipeline measure --runner quafel.runner.MyParallelRunner
kedro run --pipeline combine
kedro run --pipeline visualize
```

Here, only the pipeline `measure` will utilize multiprocessing and the rest will run single process.
We recommend this approach since there is no advantage by running the other pipelines in parallel as well.
Of course, you can run the `measure` pipeline in a single process as well by omitting the `--runner` option.

For details on the output, see the [Data Structure Section](#floppy_disk-data-structure).

Note that if you want to re-run e.g. the `visualize` pipeline, you have to re-run the `prepare` pipeline as well.
This is because intermediate data containing information about the partitions is being deleted after the `visualize` pipeline of an experimant successfully ran.
This constraint will be removed in future releases.

<details>
<summary>:construction: only:</summary>
Checkout the pre-defined VSCode tasks if you want to develop on the project.
</details>

## Configuration :wrench:

### Tweaking the Partitions

Circuits are being generated in the ```data_generation``` namespace of the project.
To adjust the number of qubits, depth of the circuit, enabled frameworks and more, checkout [conf/base/parameters/data_generation.yml](/conf/base/parameters/data_generation.yml).

### Tweaking the Execution behaviour

Everything related to executing the circuits and time measurments is contained in the ```data_science``` namespace.
Head to [conf/base/parameters/data_science.yml](/conf/base/parameters/data_science.yml) to specify a framework and set e.g. the number of evaluations.

### Tweaking the Visualization

By now, there is no specific Kedro-style configuration.
The generated plots can be adjusted using the `design` class located in [src/quafel/pipelines/visualization/nodes.py](src/quafel/pipelines/visualization/nodes.py).
Propagating these settings to a `.yml` file is on the agenda!

### Pipeline :eyeglasses:

You can actually see what's going on by running
```
poetry run kedro-viz
```
which will open a browser with [kedro-viz](https://github.com/kedro-org/kedro-viz) showing the pipeline.

## Data Structure :floppy_disk:

- [data/01_raw](data/01_raw):
  - **Versioned** [Evaluation Matrix](data/01_raw/dataset.json) containing all valid values for ```frameworks```, ```qubits```, ```depths```, and ```shots``` as specified in the [data_generation.yml](conf/base/parameters/data_generation.yml) file.
- [data/02_intermediate](data/02_intermediate):
  - Evaluation partitions split into single ```.csv``` files.
  - The number of partitions depend on the configuration.
- [data/03_qasm_circuits](data/03_qasm_circuits/):
  - A QASM circuit for each partition.
- [data/04_execution_results](data/04_execution_results/):
  - Simulator results of the job with the corresponding id.
  - Result formats are unified as a dictionary with the keys containing the binary bit representation of the measured qubit and the normalized counts as values.
  - Results are zero padded, so it is ensured that also state combinations with $0$ probability are represented.
- [data/05_execution_durations](data/05_execution_durations/):
  - Duration for the simulation of the job with the corresponding id
  - Duration is only measured using `perf_counter` and `process_time`
- [data/06_evaluations_combined](data/06_evaluations_combined/):
  - **Versioned** dataset containing the combined information of both, the input parameters (```framework```, ```qubits```, ```depth```, ```shots```), the measured duration and the simulator results
- [data/07_reportings](data/07_reporting):
  - **Versioned** dataset with the ```.json``` formatted ploty heatmaps
  - The data in this folder is named by the framework and the fixed parameter. E.g. when the number of ```qubits``` is plotted against the ```shots``` and the ```qiskit_fw``` is being used to simulate a circuit of ```depth``` $3$, the filename would be ```qiskit_fw_depth_3```.
- [data/08_print](data/07_reporting):
  - Print-ready output of the visualization pipeline in `pdf` and `png` format.

Note that all datasets that are not marked as "**versioned**" will be overwritten on the next run!

## Adding new frameworks :heavy_plus_sign:

New frameworks can easily be added by editing the [frameworks.py](src/quafel/pipelines/data_science/frameworks.py) file.
Frameworks are defined by classes following the ```NAME_fw``` naming template where ```NAME``` should be replaced by the framework to be implemented.
Later, the framework can be selected using the class name.
The constructor takes the ```qasm_circuit``` and the number of shots ```n_shots``` as parameter inputs.

The class must contain a method
```python
def execute(self) -> None:
  ...
```
which should be a minimal call to the frameworks simulator.
The output of the simulator can be stored in a class variable, e.g. ```self.result```.

This result can then be accessed in the second required method 
```python
def get_result(self) -> Dict[str, float]:
  ...
```
Here, the simulator output can be post-processed so that a dictionary with bitstring representations for the measured qubits as keys and the normalised counts as values is returned.
This dictionary is required to contain all combinations of bitstrings that result from:

```python
bitstrings = [format(i, f"0{self.n_qubits}b") for i in range (2**self.n_qubits)]
```
