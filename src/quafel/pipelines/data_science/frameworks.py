import re

import pennylane as qml
from pennylane import numpy as np

import qiskit as q
from qiskit.circuit.random import random_circuit
import numpy as np
from qiskit.quantum_info import Operator

import qibo

import cirq
from cirq.contrib.qasm_import import circuit_from_qasm

import config
import utils

import time


# class duration_framework:
        
#     def time_measurement(self, shots):
#         start_time = time.time()
#         self.execute(shots)
#         return time.time() - start_time

#     @classmethod
#     def from_config(cls):
#         return cls(
#             config.seed,
#             config.evals,
#             config.qubits,
#             config.depth,
#             config.consistent_circuit,
#         )


class pennylane_fw():
    def __init__(self, qasm_circuit, n_shots):
        circuit = self.convert_circuit(qasm_circuit)
        self.backend = qml.device("default.qubit", wires=len(circuit.wires), shots=n_shots)

        self.qc = qml.QNode(circuit, self.backend)

    def convert_circuit(qasm_circuit):
        qml.from_qasm(qasm_circuit)
        return [qml.expval(qml.PauliZ(i)) for i in range(qml.AllWires)]

    def execute(self):
        result = self.qc()
        return result

class qiskit_fw():
    def __init__(self, qasm_circuit, n_shots):
        if n_shots is None:
            self.backend = q.Aer.get_backend("statevector_simulator")
        else:
            self.backend = q.Aer.get_backend("qasm_simulator")

        self.qc = q.QuantumCircuit.from_qasm_str(qasm_circuit)
        self.n_shots = n_shots

    def execute(self):
        result = q.execute(self.qc, backend=self.backend, shots=self.n_shots).result()
        return result

# class duration_real(duration_qiskit):
#     def __init__(self, *args, **kwargs):
#         # reading token for interfacing ibm qcs
#         import ibmq_access

#         print(
#             f"Signing in to IBMQ using token {ibmq_access.token[:10]}****, hub {ibmq_access.hub}, group {ibmq_access.group} and project {ibmq_access.project}"
#         )
#         self.provider = q.IBMQ.enable_account(
#             token=ibmq_access.token,
#             hub=ibmq_access.hub,
#             group=ibmq_access.group,
#             project=ibmq_access.project,
#         )
#         self.backend = self.provider.get_backend(config.real_backend)

#         super().__init__(*args, **kwargs)

#     def generate_circuit(self, shots):
#         if self.consistent_circuit == False:
#             self._generate_qiskit_circuit(shots)
#         else:
#             self.qcs = []

#             if shots is None:
#                 return  # just return in case no shots are specified

#             for e in range(self.evals):
#                 # welchen Wert haben die qubits am Anfang?
#                 qasm_circuit = utils.get_random_qasm_circuit(
#                     self.qubits, self.depth, self.seed
#                 )
#                 qc = q.QuantumCircuit.from_qasm_str(qasm_circuit)
#                 # warum wird hier schon gemessen?
#                 self.qcs.append(qc)

#     def _generate_qiskit_circuit(self, shots):
#         self.qcs = []

#         if shots is None:
#             return  # just return in case no shots are specified

#         for e in range(self.evals):
#             # welchen Wert haben die qubits am Anfang?
#             qc = random_circuit(
#                 self.qubits, self.depth, max_operands=3, measure=True, seed=self.seed
#             )
#             # warum wird hier schon gemessen?
#             self.qcs.append(qc)

#     def execute(self, shots):
#         if self.qcs.__len__() == 0:
#             return 0
#         result = q.execute(self.qcs, backend=self.backend, shots=shots).result()

#         duration = result._metadata["time_taken"]

#         return duration

#     def time_measurement(self, shots):
#         return self.execute(shots)


# class duration_matrix(duration_framework):
#     def generate_circuit(self, shots):
#         self.qcs = []
#         for _ in range(self.evals):
#             qasm_circuit = utils.get_random_qasm_circuit(
#                 self.qubits, self.depth, self.seed, measure=False
#             )
#             qc = q.QuantumCircuit.from_qasm_str(qasm_circuit)
#             matrix = Operator(qc)
#             self.qcs.append(matrix)

#     def execute(self, shots):
#         # keine circuits sondern fertige Matrizen
#         for matrix in self.qcs:
#             statevector = np.array(matrix)[:, 0]
#             probabilities = np.abs((statevector) ** 2)
#             if shots is not None:
#                 np.random.choice(len(probabilities), shots, p=probabilities)


# class duration_cirq(duration_framework):
#     def generate_circuit(self, shots):
#         if self.consistent_circuit == False:
#             self._generate_cirq_circuit()
#         else:
#             self.qcs = []
#             for i in range(self.evals):
#                 qasm_circuit = utils.get_random_qasm_circuit(
#                     self.qubits, self.depth, self.seed
#                 )
#                 circuit = circuit_from_qasm(qasm_circuit)
#                 circuit.append(
#                     cirq.measure(
#                         cirq.NamedQubit.range(self.qubits, prefix=""), key="result"
#                     )
#                 )
#                 self.qcs.append(circuit)
#             self.simulator = cirq.Simulator()

#     def _generate_cirq_circuit(self):
#         self.qcs = []
#         for i in range(self.evals):
#             circuit = cirq.testing.random_circuit(
#                 qubits=self.qubits,
#                 n_moments=self.depth,
#                 random_state=self.seed,
#                 op_density=0.5,
#             )
#             circuit.append(
#                 cirq.measure(
#                     cirq.NamedQubit.range(self.qubits, prefix=""), key="result"
#                 )
#             )
#             self.qcs.append(circuit)
#         self.simulator = cirq.Simulator()

#     def execute(self, shots):
#         for i in self.qcs:
#             if shots is None:
#                 self.simulator.simulate(i)
#             else:
#                 self.simulator.run(i, repetitions=shots)


# class duration_qibo(duration_framework):
#     def generate_circuit(self, shots):
#         if self.consistent_circuit == False:
#             raise NotImplementedError
#         else:
#             self.qcs = []

#             if shots is None:
#                 return
#             self.backend = qibo.get_backend()

#             for e in range(self.evals):
#                 # welchen Wert haben die qubits am Anfang?
#                 qasm_circuit = utils.get_random_qasm_circuit(
#                     self.qubits, self.depth, self.seed
#                 )

#                 # this is super hacky, but the way qibo parses the QASM string
#                 # does not deserve better.
#                 def qasm_conv(match: re.Match) -> str:
#                     denominator = float(match.group()[1:])
#                     return f"*{1/denominator}"

#                 qasm_circuit = re.sub(
#                     r"/\d*", qasm_conv, qasm_circuit, flags=re.MULTILINE
#                 )

#                 qc = qibo.models.Circuit.from_qasm(qasm_circuit)
#                 # warum wird hier schon gemessen?
#                 self.qcs.append(qc)

#             self.states = [np.random.random(2**self.qubits) for i in range(5)]

#     def _generate_qibo_circuit(self, shots):
#         pass

#     def execute(self, shots):
#         if self.qcs.__len__() == 0:
#             return 0

#         for i in self.qcs:
#             qibo.set_threads(1)
#             # execute in parallel
#             qibo.parallel.parallel_execution(i, self.states, processes=10)
