
# Data Structure

- [data/01_raw](data/01_raw):
  - **Versioned** [Evaluation Matrix](data/01_raw/dataset.json) containing all valid values for ```frameworks```, ```qubits```, ```depths```, and ```shots``` as specified in the [data_generation.yml](conf/base/parameters/data_generation.yml) file.
- [data/02_intermediate](data/02_intermediate):
  - Evaluation partitions split into single ```.csv``` files.
  - The number of partitions depend on the configuration.
- [data/03_qasm_circuits](data/03_qasm_circuits/):
  - A QASM circuit for each partition.
- [data/04_measures](data/04_measures/):
  - Entangling capability and expressibility of each generated circuit
  - Calculation according to [Sim et al. - Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms](https://arxiv.org/abs/1905.10876)
  - Statevectors of circuits are cached (`./.cache/` folder) based on the md5 hash of provided circuit to speedup calculation
- [data/05_execution_results](data/05_execution_results/):
  - Simulator results of the job with the corresponding id.
  - Result formats are unified as a dictionary with the keys containing the binary bit representation of the measured qubit and the normalized counts as values.
  - Results are zero padded, so it is ensured that also state combinations with $0$ probability are represented.
- [data/06_execution_durations](data/06_execution_durations/):
  - Duration for the simulation of the job with the corresponding id
  - Duration is only measured using `perf_counter` and `process_time`
- [data/07_evaluations_combined](data/07_evaluations_combined/):
  - **Versioned** dataset containing the combined information of both, the input parameters (```framework```, ```qubits```, ```depth```, ```shots```), the measured duration and the simulator results
- [data/08_reportings](data/08_reportings):
  - **Versioned** dataset with the ```.json``` formatted ploty heatmaps
  - The data in this folder is named by the framework and the fixed parameter. E.g. when the number of ```qubits``` is plotted against the ```shots``` and the ```qiskit_fw``` is being used to simulate a circuit of ```depth``` $3$, the filename would be ```qiskit_fw_depth_3```.
- [data/09_print](data/09_print):
  - Print-ready output of the visualization pipeline in `pdf` and `png` format.

Note that all datasets that are not marked as "**versioned**" will be overwritten on the next run!

