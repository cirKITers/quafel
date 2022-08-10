#import pennylane as qml
#from pennylane import numpy as np
#import qiskit as q
from qiskit.circuit.random import random_circuit
from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
import cirq
#import config

def get_random_qasm_circuit(qubits, depth, seed):
    qc = random_circuit(qubits, depth, max_operands=3, measure=True, seed=seed)
    qc = transpile(qc, basis_gates=['u2', 'u3', 'cx', 'id', 'x', 'y', 'z', 'h', 's', 't', 'rx', 'ry', 'rz', 'cx', 'cy', 'cz', 'ch', 'swap', 'ccx', 'cswap'])
    return qc.qasm()