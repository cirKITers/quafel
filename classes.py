import pennylane as qml
from pennylane import numpy as np

import qiskit as q
from qiskit.circuit.random import random_circuit

import cirq
from cirq.contrib.qasm_import import circuit_from_qasm

import config
import utils


class initialize:
    def __init__(self, seed, evals, qubits, depth, consistent_circuit):
        self.seed = seed
        self.evals = evals
        self.qubits = qubits
        self.depth = depth
        self.consistent_circuit = consistent_circuit

    @classmethod
    def from_config(cls):
        return cls(
            config.seed,
            config.evals,
            config.qubits,
            config.depth,
            config.consistent_circuit,
        )


class duration_pennylane(initialize):
    def generate_circuit(self, shots):
        if self.consistent_circuit == False:
            self._generate_pennylane_circuit(shots)

        else:
            self.dev = qml.device("default.qubit", wires=self.qubits, shots=shots)

            def create_circuit():
                circuit = qml.from_qasm(
                    utils.get_random_qasm_circuit(self.qubits, self.depth, self.seed)
                )
                return [qml.expval(qml.PauliZ(i)) for i in range(self.qubits)]

            qnode = qml.QNode(create_circuit, self.dev)
            self.qcs = []
            for i in range(self.evals):
                self.qcs.append(qnode)

    def _generate_pennylane_circuit(self, shots):
        self.dev = qml.device("default.qubit", wires=self.qubits, shots=shots)
        self.w = np.random.rand(self.depth, self.qubits)

        def create_circuit(w):
            qml.RandomLayers(weights=w, wires=range(self.qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.qubits)]

        qnode = qml.QNode(create_circuit, self.dev)
        w = self.w
        self.qcs = []
        for i in range(self.evals):
            self.qcs.append(qnode)

    def execute(self, shots):
        if self.consistent_circuit == False:
            self.__execute_specific_circuit()
        else:
            for circuit in self.qcs:
                circuit()

    def __execute_specific_circuit(self):
        for circuit in self.qcs:
            circuit(self.w)


class duration_qiskit(initialize):
    def generate_circuit(self, shots):
        if self.consistent_circuit == False:
            self._generate_qiskit_circuit(shots)
        else:
            if shots == None:
                self.backend = q.Aer.get_backend("statevector_simulator")
            else:
                self.backend = q.Aer.get_backend("qasm_simulator")

            self.qcs = []
            for e in range(self.evals):
                # welchen Wert haben die qubits am Anfang?
                qasm_circuit = utils.get_random_qasm_circuit(
                    self.qubits, self.depth, self.seed
                )
                qc = q.QuantumCircuit.from_qasm_str(qasm_circuit)
                # warum wird hier schon gemessen?
                self.qcs.append(qc)

    def _generate_qiskit_circuit(self, shots):
        if shots == None:
            self.backend = q.Aer.get_backend("statevector_simulator")
        else:
            self.backend = q.Aer.get_backend("qasm_simulator")

        self.qcs = []
        for e in range(self.evals):
            # welchen Wert haben die qubits am Anfang?
            qc = random_circuit(
                self.qubits, self.depth, max_operands=3, measure=True, seed=self.seed
            )
            # warum wird hier schon gemessen?
            qc.measure_all()
            self.qcs.append(qc)

    def execute(self, shots):
        result = q.execute(self.qcs, backend=self.backend, shots=shots).result()


class duration_cirq(initialize):
    def generate_circuit(self, shots):
        if self.consistent_circuit == False:
            self._generate_cirq_circuit()
        else:
            self.qcs = []
            for i in range(self.evals):
                qasm_circuit = utils.get_random_qasm_circuit(
                    self.qubits, self.depth, self.seed
                )
                circuit = circuit_from_qasm(qasm_circuit)
                circuit.append(
                    cirq.measure(
                        cirq.NamedQubit.range(self.qubits, prefix=""), key="result"
                    )
                )
                self.qcs.append(circuit)
            self.simulator = cirq.Simulator()

    def _generate_cirq_circuit(self):
        self.qcs = []
        for i in range(self.evals):
            circuit = cirq.testing.random_circuit(
                qubits=self.qubits,
                n_moments=self.depth,
                random_state=self.seed,
                op_density=0.5,
            )
            circuit.append(
                cirq.measure(
                    cirq.NamedQubit.range(self.qubits, prefix=""), key="result"
                )
            )
            self.qcs.append(circuit)
        self.simulator = cirq.Simulator()

    def execute(self, shots):
        for i in self.qcs:
            if shots == None:
                result = self.simulator.simulate(i)
            else:
                result = self.simulator.run(i, repetitions=shots)
