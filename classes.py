import pennylane as qml
from pennylane import numpy as np
import qiskit as q
from qiskit.circuit.random import random_circuit
import cirq
import config

class duration_pennylane:   
    def __init__(self):
        self.seed = config.seed
        self.evals = config.evals
        self.qubits = config.qubits
        self.depth = config.depth
        
    def generate_circuit(self, shots):
        self.dev = qml.device("default.qubit", wires=self.qubits, shots=shots)
        self.w = np.random.rand(self.depth, self.qubits)

        """    
        @qml.qnode(self.dev)        
        def create_circuit(w):
            qml.RandomLayers(weights=w, wires=range(self.qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.qubits)]
        w = self.w

        for i in self.evals:
            self.qcs.append(create_circuit(w))
        return create_circuit(w) 
        """             
        """ def create_circuit(w):
            qml.RandomLayers(weights=w, wires=range(self.qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.qubits)]
        qnode = qml.QNode(create_circuit, self.dev)
        w = self.w
        self.qcs = []
        for i in range(self.evals):
            self.qcs.append(create_circuit(w))
        return qnode(w) """
        
        def create_circuit(w):
            qml.RandomLayers(weights=w, wires=range(self.qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.qubits)]
        qnode = qml.QNode(create_circuit, self.dev)
        w = self.w
        self.qcs = []
        for i in range(self.evals):
            self.qcs.append(qnode)
        
    def execute(self, shots):
        for i in self.qcs:
            i(self.w)


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


class duration_cirq():
    def __init__(self):
        self.seed = config.seed
        self.evals = config.evals
        self.qubits = config.qubits
        self.depth = config.depth

    def generate_circuit(self, shots):
        self.qcs = []
        for i in range(self.evals):
            self.circuit = cirq.testing.random_circuit(qubits = self.qubits, n_moments = self.depth, random_state = self.seed, op_density=0.5)
            self.circuit.append(cirq.measure(cirq.NamedQubit.range(self.qubits, prefix=''),key='result'))
            self.qcs.append(self.circuit)        
        self.simulator = cirq.Simulator()
        
    def execute(self, shots):
        for i in self.qcs: 
            if shots == None:
                result = self.simulator.simulate(i)
            else:
                result = self.simulator.run(i, repetitions=shots)

