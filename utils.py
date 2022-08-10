#import pennylane as qml
#from pennylane import numpy as np
#import qiskit as q
from qiskit.circuit.random import random_circuit
from qiskit.circuit import QuantumCircuit
import cirq
#import config

def get_random_qasm_circuit(qubits, depth, seed):
    qc = random_circuit(qubits, depth, max_operands=3, measure=True, seed=seed)
    return qc.qasm()