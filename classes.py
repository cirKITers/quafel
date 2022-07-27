import pennylane as qml
from pennylane import numpy as np

import qiskit as q
from qiskit.circuit.random import random_circuit

import config

class duration_pennylane:   
    def __init__(self):
        self.shots_list = config.shots_list
        self.seed = config.seed
        self.evals = config.evals
        self.qubits = config.qubits
        self.depth = config.depth
        
        

    def generate_circuit(self, shots):
        self.dev = qml.device("default.qubit", wires=self.qubits, shots=shots)
        self.w = np.random.rand(self.depth, self.qubits)

        @qml.qnode(self.dev)        
        def create_circuit(w):
            qml.RandomLayers(weights=w, wires=range(self.qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.qubits)]
        w = self.w
        return create_circuit(w)
        """ #@qml.qnode(self.dev)        
        def create_circuit(w):
            qml.RandomLayers(weights=w, wires=range(self.qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.qubits)]
        qnode = qml.QNode(create_circuit, self.dev)
        w = self.w
        return qnode(w) """
   
    
    
    def execute(self, shots):
        for _ in range(self.evals):
            self.generate_circuit(shots)

class duration_qiskit:   
    def __init__(self):
        self.shots_list = config.shots_list
        self.seed = config.seed
        self.evals = config.evals
        self.qubits = config.qubits
        self.depth = config.depth

    def generate_circuit(self, shots):
        if shots == None:
            self.backend = q.Aer.get_backend("statevector_simulator")
        else:
            self.backend = q.Aer.get_backend("qasm_simulator")

        self.qcs = []
        for e in range(self.evals):
            # welchen Wert haben die qubits am Anfang?
            qc = random_circuit(self.qubits, self.depth, max_operands=3, measure=True, seed=self.seed)
            # warum wird hier schon gemessen?
            qc.measure_all()

            self.qcs.append(qc)
        

    def execute(self, shots):
        result = q.execute(self.qcs, backend=self.backend, shots=shots).result() 
