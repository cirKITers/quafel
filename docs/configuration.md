# Configuration

## Tweaking the Partitions

Circuits are being generated in the ```data_generation``` namespace of the project.
To adjust the number of qubits, depth of the circuit, enabled frameworks and more, checkout [conf/base/parameters/data_generation.yml](/conf/base/parameters/data_generation.yml).

Here you can adjust the following parameters:
- `seed`: Used in the circuit generation method to sample random gates
- `samples_per_parameter`: for expressibility and entangling capability measures
- `haar_samples_per_qubit`: for expressibility and entangling capability measures
- `min_[qubits/depth/shots]`: lowest number of qubits/ circuit depth/ shots used for generating partitions
- `max_[qubits/depth/shots]`: highest number of qubits/ circuit depth/ shots used for generating partitions
- `[qubits/depth/shots]_increment`: steps in which the range specified by min/max value will be iterated
- `[qubits/depth/shots]_type`: type of the increment (e.g. `exp2` or `linear`)
  

## Tweaking the Execution behaviour

Everything related to executing the circuits and time measurments is contained in the ```data_science``` namespace.
Head to [conf/base/parameters/data_science.yml](/conf/base/parameters/data_science.yml) to specify a framework and set e.g. the number of evaluations.

## Tweaking the Visualization

By now, there is no specific Kedro-style configuration.
The generated plots can be adjusted using the `design` class located in [src/quafel/pipelines/visualization/nodes.py](src/quafel/pipelines/visualization/nodes.py).
Propagating these settings to a `.yml` file is on the agenda!

## Pipeline :eyeglasses:

You can actually see what's going on by running
```
poetry run kedro-viz
```
which will open a browser with [kedro-viz](https://github.com/kedro-org/kedro-viz) showing the pipeline.
