# Adding New Frameworks

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
