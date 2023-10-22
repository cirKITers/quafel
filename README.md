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
poetry kedro run --pipeline "prepare"
```
followed by
```
poetry kedro run
```
(or omit poetry if you're using classical venvs) and a default pipeline should run.

Note that is required to always run the "prepare" pipeline in advance to any actual processing pipeline.
This is because of the current implementation relies on dynamically created nodes that are depending on the configuration and therefore requiring two separate pipeline executions.

In summary, the following pipelines exist:
- "prepare" : generates all possible combinations of configurations based on the current parameter set
- "measure" : performs the actual time measurement by executing experiments for each of the previously generated configuration
- "ctmeasure" : continous a previous time measurement
- "visualize" : gathers all the experiment results and generates some nice plots

The "default" pipeline covers "measure" and "visualize".
If you want to run them separately execute
```
poetry kedro run --pipeline "measure"
```
and
```
poetry kedro run --pipeline "visualize"
```
after running the "prepare" pipeline.

This project can take advantage of multiprocessing using [Dask](dask.org/) to evaluate numerous combinations of *qubits*, *depths* and *shots*.
To enable this, you can run
```
poetry run kedro run --pipeline "measure" --env dask --runner quafel.runner.DaskRunner
```
which will calculate the duration and result for each configuration.
See [Dask Setup](#runner-dask-setup) for detail on this.
For details on the output, see the [Data Structure Section](#floppy_disk-data-structure).


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
  - simulator results of the job with the corresponding id.
  - result formats are unified as a dictionary with the keys containing the binary bit representation of the measured qubit and the normalized counts as values.
  - results are zero padded, so it is ensured that also state combinations with $0$ probability are represented.
- [data/05_execution_durations](data/05_execution_durations/):
  - duration for the simulation of the job with the corresponding id.
  - duration is only measured for the execution of the simulator
  - combining and post-processing results (to obtain the dictionary representation) is not involved
- [data/06_evaluations_combined](data/06_evaluations_combined/):
  - **Versioned** dataset containing the combined information of both, the input parameters (```framework```, ```qubits```, ```depth```, ```shots```), the measured duration and the simulator results
- [data/07_reportings](data/07_reporting):
  - **Versioned** dataset with the ```.json``` formatted ploty heatmaps
  - The data in this folder is named by the framework and the fixed parameter. E.g. when the number of ```qubits``` is plotted against the ```shots``` and the ```qiskit_fw``` is being used to simulate a circuit of ```depth``` $3$, the filename would be ```qiskit_fw_depth_3```.

Note that all datasets that are not marked as "**versioned**" will be overwritten on the next run!

## :construction: Adding new frameworks

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

## :runner: Dask Setup

When running
```
poetry run kedro run --pipeline "measure" --env dask --runner quafel.runner.DaskRunner
```
without any additional configuration, Kedro creates Dask scheduler and also four worker nodes.
This behavior can be controlled in [conf/dask/parameters.yml](conf/dask/parameters.yml).
Setting the address parameter will cause Kedro trying to connect to an existing scheduler at the specified address.
You create scheduler and $N$ workers by running `.vscode/spawn_n_workers.sh -N` from the root folder of the project.
Alternatively set `n_workers` in [conf/dask/parameters.yml](conf/dask/parameters.yml) and comment out `address` to specify the number of workers that kedro should spawn.