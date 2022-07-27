# idea: one class (and file) for each model
# maybe add an abstract class Model later on to force certain methods
# find useful methods that are needed in all models

class Qiskit_Model:
    def __init__(self) -> None:
        pass

    def execute_circuit(self):
        pass


class Pennylane_Model:
    def __init__(self) -> None:
        pass

    def execute_circuit(self):
        pass


# main file to run experiment. Only one if-clause / one line to switch between models
# maybe add arguments later on

FRAMEWORK = "qiskit"
MODE = "DEPTH"

if FRAMEWORK == "qiskit":
    model = Qiskit_Model()
elif FRAMEWORK == "pennylane":
    model = Pennylane_Model()

def depth(model, shot_list, depth):
    # something like this
    for _ in range(shot_list):
        for _ in range(depth):
            model.execute_circuit()

def qubits():
    # ...
    pass

