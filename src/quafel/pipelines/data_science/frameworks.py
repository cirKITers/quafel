import time
import re
import numpy as np

import dask.array as da

import pennylane as qml

import qiskit
from qiskit.quantum_info import Operator

import qulacs
from qulacs import QuantumCircuit as QualcsQuantumCircuit
from qulacs.gate import DenseMatrix, Identity

import qibo

import qrisp

import cirq
from cirq.contrib.qasm_import import circuit_from_qasm


from typing import Dict, List


def calculate_n_qubits_from_qasm(qasm_string):
    return int(
        re.findall(r"qreg q\[(?P<n_qubits>\d*)\]", qasm_string)[0]
    )  # TODO: improvement wanted


class test_fw:
    time_const = 1e-9
    constant_sleep = True
    load = True

    def __init__(self, qasm_circuit, n_shots):
        self.n_qubits = calculate_n_qubits_from_qasm(qasm_circuit)

        self.depth = int(
            qasm_circuit[qasm_circuit.find("\nqreg q[") + 8]
        )  # TODO: improvement wanted

        self.shots = n_shots

    def execute(self) -> None:
        if self.constant_sleep:
            if self.load:
                # Following https://tutorial.dask.org/02_array.html#Dask-array-version
                xd = da.random.normal(10, 0.1, size=(300, 300), chunks=(30, 30))
                yd = xd.mean(axis=0)
                yd.compute()
            else:
                time.sleep(0.01)
        else:
            time.sleep(
                self.time_const * self.shots * self.depth**2 * self.n_qubits**3
            )

    def get_result(self) -> Dict[str, float]:
        counts = {}

        for i in range(2**self.n_qubits):
            bitstring = format(i, f"0{self.n_qubits}b")
            counts[bitstring] = 0

        return counts


class pennylane_fw:
    def __init__(self, qasm_circuit, n_shots):
        self.n_qubits = calculate_n_qubits_from_qasm(qasm_circuit)

        self.n_shots = n_shots

        if self.n_qubits <= 20:
            self.backend = qml.device(
                "default.qubit",
                wires=range(self.n_qubits),
                shots=self.n_shots,
                max_workers=None,  # restrict subworkers to 1 to prevent issues with MP https://docs.pennylane.ai/en/stable/code/api/pennylane.devices.default_qubit.DefaultQubit.html # noqa
            )
        else:  # recommended to be used for > 20 qubits
            self.backend = qml.device(
                "lightning.qubit", wires=range(self.n_qubits), shots=self.n_shots
            )

        # Pennylane does not support measurements at the moment
        qasm_circuit_wo_measurement = re.sub(r"measure.*;\n", "", qasm_circuit)
        self.qml_qasm = qml.from_qasm(qasm_circuit_wo_measurement)

        @qml.qnode(self.backend)
        def circuit():
            self.qml_qasm()
            return qml.counts()

        self.qc = circuit

    def execute(self) -> None:
        self.result = self.qc()

    def get_result(self) -> Dict[str, float]:
        counts = self.result

        for i in range(2**self.n_qubits):
            bitstring = format(i, f"0{self.n_qubits}b")
            if bitstring not in counts.keys():
                counts[bitstring] = 0

            else:
                counts[bitstring] = counts[bitstring] / self.n_shots

        return counts


class qiskit_fw:
    def __init__(self, qasm_circuit, n_shots):
        # self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.backend = qiskit.Aer.get_backend("aer_simulator")

        self.n_qubits = calculate_n_qubits_from_qasm(qasm_circuit)

        self.qc = qiskit.QuantumCircuit.from_qasm_str(qasm_circuit)
        self.n_shots = n_shots
        self.result = None

    def execute(self) -> None:
        self.result = qiskit.execute(self.qc, backend=self.backend, shots=self.n_shots)

    def get_result(self) -> Dict[str, float]:
        counts = self.result.result().get_counts()

        for i in range(2**self.n_qubits):
            bitstring = format(i, f"0{self.n_qubits}b")
            if bitstring not in counts.keys():
                counts[bitstring] = 0

        return counts


class qrisp_fw:
    def __init__(self, qasm_circuit, n_shots):
        self.n_qubits = calculate_n_qubits_from_qasm(qasm_circuit)

        self.qc = qrisp.QuantumCircuit.from_qasm_str(qasm_circuit)
        self.n_shots = n_shots
        self.result = None

    def execute(self) -> None:
        self.result = self.qc.run(shots=self.n_shots)

    def get_result(self) -> Dict[str, float]:
        counts = self.result

        for i in range(2**self.n_qubits):
            bitstring = format(i, f"0{self.n_qubits}b")
            if bitstring not in counts.keys():
                counts[bitstring] = 0

        return counts


class numpy_fw:
    def __init__(self, qasm_circuit, n_shots):
        self.n_shots = n_shots
        cicruit = qiskit.QuantumCircuit.from_qasm_str(qasm_circuit)
        cicruit.remove_final_measurements()
        self.n_qubits = calculate_n_qubits_from_qasm(qasm_circuit)
        self.qc = Operator(cicruit)

    def execute(self) -> None:
        # keine circuits sondern fertige Matrizen
        statevector = np.array(self.qc)[:, 0]
        probabilities = np.abs((statevector) ** 2)
        if self.n_shots is not None:
            self.result = np.random.choice(
                len(probabilities), self.n_shots, p=probabilities
            )
        else:
            self.result = probabilities

    def get_result(self) -> Dict[str, float]:
        counts = {}

        # TODO verification needed!
        for i in range(2**self.n_qubits):
            bitstring = format(i, f"0{self.n_qubits}b")
            c = np.count_nonzero(self.result == i)
            counts[bitstring] = c

        return counts


class cirq_fw:
    def __init__(self, qasm_circuit, n_shots):
        self.n_shots = n_shots

        self.n_qubits = calculate_n_qubits_from_qasm(qasm_circuit)

        self.qc = circuit_from_qasm(qasm_circuit)

        self.qc.append(
            cirq.measure(cirq.NamedQubit.range(self.n_qubits, prefix=""), key="result")
        )
        self.backend = cirq.Simulator()

    def execute(self) -> None:
        if self.n_shots is None:
            self.result = self.backend.simulate(self.qc)
        else:
            self.result = self.backend.run(self.qc, repetitions=self.n_shots)

    def get_result(self) -> Dict[str, float]:
        counts = {}

        # TODO verification needed!
        for i in range(2**self.n_qubits):
            bitstring = format(i, f"0{self.n_qubits}b")

            mask = None
            for j in range(self.n_qubits):
                if mask is None:
                    mask = self.result.data[f"c_{j}"] == int(bitstring[j])
                else:
                    mask &= self.result.data[f"c_{j}"] == int(bitstring[j])
            result = len(self.result.data[mask])

            counts[bitstring] = result / self.n_shots

        return counts


class qibo_fw:
    def __init__(self, qasm_circuit, n_shots):
        qibo.set_backend("numpy")
        self.backend = qibo.get_backend()
        self.n_shots = n_shots

        self.n_qubits = calculate_n_qubits_from_qasm(qasm_circuit)

        # this is super hacky, but the way qibo parses the QASM string
        # does not deserve better.
        def qasm_conv(match: re.Match) -> str:
            denominator = float(match.group()[1:])
            return f"*{1/denominator}"

        qasm_circuit = re.sub(r"/\d*", qasm_conv, qasm_circuit, flags=re.MULTILINE)

        self.qc = qibo.models.Circuit.from_qasm(qasm_circuit)

    def execute(self) -> None:
        self.result = self.qc(nshots=self.n_shots)

    def get_result(self) -> Dict[str, float]:
        counts = dict(self.result.frequencies(binary=True))

        # TODO verification needed!
        for i in range(2**self.n_qubits):
            bitstring = format(i, f"0{self.n_qubits}b")
            if bitstring not in counts.keys():
                counts[bitstring] = 0

        return counts


class qulacs_fw:
    def __init__(self, qasm_circuit, n_shots):
        self.n_qubits = calculate_n_qubits_from_qasm(qasm_circuit)
        # removing measurment operation...
        qasm_circuit = re.sub("\\nmeasure .*;", "", qasm_circuit)
        # ... and creg operation as they are not supported by qulacs
        # see `convert_QASM_to_qulacs_circuit` for details.
        qasm_circuit = re.sub("\\ncreg .*;", "", qasm_circuit)
        self.qc = self.convert_QASM_to_qulacs_circuit(qasm_circuit.split("\n"))
        self.n_shots = n_shots
        self.result = None

    def execute(self) -> None:
        state = qulacs.QuantumState(self.n_qubits)
        self.qc.update_quantum_state(state)
        self.result = state.sampling(sampling_count=self.n_shots)

    def get_result(self) -> Dict[str, float]:
        counts = {}
        for i in range(2**self.n_qubits):
            bitstring = format(i, f"0{self.n_qubits}b")
            if i not in self.result:
                counts[bitstring] = 0
            else:
                counts[bitstring] = sum(c == i for c in self.result)

        return counts

    def convert_QASM_to_qulacs_circuit(
        self, input_strs: List[str], *, remap_remove: bool = False
    ) -> QualcsQuantumCircuit:
        """
        Copied from https://github.com/qulacs/qulacs/blob/2750414e8e1ebc61064207158dd8b0e618708ad4/pysrc/qulacs/converter/qasm_converter.py#L120 # noqa
        Only difference is the missing bracket for the u2 parsing

        convert QASM List[str] to qulacs QuantumCircuit.

        constraints: qreg must be named q, and creg cannot be used.
        """
        FIXED_POINT_PATTERN = r"[+-]?\d+(?:\.\d*)?|\.\d+"
        FLOATING_POINT_PATTERN = r"[eE][-+]?\d+"
        GENERAL_NUMBER_PATTERN = (
            rf"(?:{FIXED_POINT_PATTERN})(?:{FLOATING_POINT_PATTERN})?"  # noqa
        )

        mapping: List[int] = []

        for instr_moto in input_strs:
            # process input string for parsing instruction.
            instr = instr_moto.lower().strip().replace(" ", "").replace("\t", "")
            if instr == "":
                continue
            if instr[0:4] == "qreg":
                matchobj = re.match(r"qregq\[(\d+)\];", instr)
                assert matchobj is not None
                ary = matchobj.groups()
                cir = QualcsQuantumCircuit(int(ary[0]))
                if len(mapping) == 0:
                    mapping = list(range(int(ary[0])))
            elif instr[0:2] == "cx":
                matchobj = re.match(r"cxq\[(\d+)\],q\[(\d+)\];", instr)
                assert matchobj is not None
                ary = matchobj.groups()
                cir.add_CNOT_gate(mapping[int(ary[0])], mapping[int(ary[1])])
            elif instr[0:2] == "cz":
                matchobj = re.match(r"czq\[(\d+)\],q\[(\d+)\];", instr)
                assert matchobj is not None
                ary = matchobj.groups()
                cir.add_CZ_gate(mapping[int(ary[0])], mapping[int(ary[1])])
            elif instr[0:4] == "swap":
                matchobj = re.match(r"swapq\[(\d+)\],q\[(\d+)\];", instr)
                assert matchobj is not None
                ary = matchobj.groups()
                cir.add_SWAP_gate(mapping[int(ary[0])], mapping[int(ary[1])])
            elif instr[0:2] == "id":
                matchobj = re.match(r"idq\[(\d+)\];", instr)
                assert matchobj is not None
                ary = matchobj.groups()
                cir.add_gate(Identity(mapping[int(ary[0])]))
            elif instr[0:2] == "xq":
                matchobj = re.match(r"xq\[(\d+)\];", instr)
                assert matchobj is not None
                ary = matchobj.groups()
                cir.add_X_gate(mapping[int(ary[0])])
            elif instr[0:2] == "yq":
                matchobj = re.match(r"yq\[(\d+)\];", instr)
                assert matchobj is not None
                ary = matchobj.groups()
                cir.add_Y_gate(mapping[int(ary[0])])
            elif instr[0:2] == "zq":
                matchobj = re.match(r"zq\[(\d+)\];", instr)
                assert matchobj is not None
                ary = matchobj.groups()
                cir.add_Z_gate(mapping[int(ary[0])])
            elif instr[0:2] == "hq":
                matchobj = re.match(r"hq\[(\d+)\];", instr)
                assert matchobj is not None
                ary = matchobj.groups()
                cir.add_H_gate(mapping[int(ary[0])])
            elif instr[0:2] == "sq":
                matchobj = re.match(r"sq\[(\d+)\];", instr)
                assert matchobj is not None
                ary = matchobj.groups()
                cir.add_S_gate(mapping[int(ary[0])])
            elif instr[0:4] == "sdgq":
                matchobj = re.match(r"sdgq\[(\d+)\];", instr)
                assert matchobj is not None
                ary = matchobj.groups()
                cir.add_Sdag_gate(mapping[int(ary[0])])
            elif instr[0:2] == "tq":
                matchobj = re.match(r"tq\[(\d+)\];", instr)
                assert matchobj is not None
                ary = matchobj.groups()
                cir.add_T_gate(mapping[int(ary[0])])
            elif instr[0:4] == "tdgq":
                matchobj = re.match(r"tdgq\[(\d+)\];", instr)
                assert matchobj is not None
                ary = matchobj.groups()
                cir.add_Tdag_gate(mapping[int(ary[0])])
            elif instr[0:2] == "rx":
                matchobj = re.match(
                    rf"rx\(({GENERAL_NUMBER_PATTERN})\)q\[(\d+)\];", instr
                )
                assert matchobj is not None
                ary = matchobj.groups()
                cir.add_RX_gate(mapping[int(ary[1])], -float(ary[0]))
            elif instr[0:2] == "ry":
                matchobj = re.match(
                    rf"ry\(({GENERAL_NUMBER_PATTERN})\)q\[(\d+)\];", instr
                )
                assert matchobj is not None
                ary = matchobj.groups()
                cir.add_RY_gate(mapping[int(ary[1])], -float(ary[0]))
            elif instr[0:2] == "rz":
                matchobj = re.match(
                    rf"rz\(({GENERAL_NUMBER_PATTERN})\)q\[(\d+)\];", instr
                )
                assert matchobj is not None
                ary = matchobj.groups()
                cir.add_RZ_gate(mapping[int(ary[1])], -float(ary[0]))
            elif instr[0:1] == "p":
                matchobj = re.match(rf"p\({GENERAL_NUMBER_PATTERN}\)q\[(\d+)\];", instr)
                assert matchobj is not None
                ary = matchobj.groups()
                cir.add_U1_gate(mapping[int(ary[1])], float(ary[0]))
            elif instr[0:2] == "u1":
                matchobj = re.match(
                    rf"u1\(({GENERAL_NUMBER_PATTERN})\)q[(\d+)];", instr
                )
                assert matchobj is not None
                ary = matchobj.groups()
                cir.add_U1_gate(mapping[int(ary[1])], float(ary[0]))
            elif instr[0:2] == "u2":
                matchobj = re.match(
                    rf"u2\(({GENERAL_NUMBER_PATTERN}),"
                    + rf"({GENERAL_NUMBER_PATTERN})\)q\[(\d+)\];",
                    instr,
                )
                assert matchobj is not None
                ary = matchobj.groups()
                cir.add_U2_gate(mapping[int(ary[2])], float(ary[0]), float(ary[1]))
            elif instr[0:2] == "u3":
                matchobj = re.match(
                    rf"u3\(({GENERAL_NUMBER_PATTERN}),"
                    + rf"({GENERAL_NUMBER_PATTERN}),"
                    + rf"({GENERAL_NUMBER_PATTERN})\)q\[(\d+)\];",
                    instr,
                )
                assert matchobj is not None
                ary = matchobj.groups()
                cir.add_U3_gate(
                    mapping[int(ary[3])], float(ary[0]), float(ary[1]), float(ary[2])
                )
            elif instr[0:1] == "u":
                matchobj = re.match(
                    rf"u\(({GENERAL_NUMBER_PATTERN}),"
                    + rf"({GENERAL_NUMBER_PATTERN}),"
                    + rf"({GENERAL_NUMBER_PATTERN})\)q\[(\d+)\];",
                    instr,
                )
                assert matchobj is not None
                ary = matchobj.groups()
                cir.add_U3_gate(
                    mapping[int(ary[3])], float(ary[0]), float(ary[1]), float(ary[2])
                )
            elif instr[0:3] == "sxq":
                matchobj = re.match(r"sxq\[(\d+)\];", instr)
                assert matchobj is not None
                ary = matchobj.groups()
                cir.add_sqrtX_gate(mapping[int(ary[0])])
            elif instr[0:5] == "sxdgq":
                matchobj = re.match(r"sxdgq\[(\d+)\];", instr)
                assert matchobj is not None
                ary = matchobj.groups()
                cir.add_sqrtXdag_gate(mapping[int(ary[0])])
            elif instr[0:11] == "densematrix":
                # Matches all matrix elements and qubit indexes
                deary = re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", instr)
                target_qubit_count = int(deary[0])
                control_qubit_count = int(deary[1])

                gate_mat = np.zeros(
                    (2**target_qubit_count, 2**target_qubit_count), dtype="complex"
                )
                bas = 2
                for i in range(2**target_qubit_count):
                    for j in range(2**target_qubit_count):
                        gate_mat[i][j] = (
                            float(deary[bas]) + float(deary[bas + 1]) * 1.0j
                        )
                        bas += 2
                control_values = []
                for i in range(control_qubit_count):
                    control_values.append(mapping[int(deary[bas])])
                    bas += 1
                terget_indexes = []
                for i in range(target_qubit_count):
                    terget_indexes.append(mapping[int(deary[bas])])
                    bas += 1

                dense_gate = DenseMatrix(terget_indexes, gate_mat)  # type:ignore
                for i in range(control_qubit_count):
                    control_index = int(deary[bas])
                    bas += 1
                    dense_gate.add_control_qubit(control_index, control_values[i])
                cir.add_gate(dense_gate)
            elif remap_remove and instr[0:4] == "//q[":
                matchobj = re.match(r"//q\[(\d+)-->q\[(\d+)\]", instr)
                assert matchobj is not None
                ary = matchobj.groups()
                if not (ary is None):
                    mapping[int(ary[0])] = int(ary[1])
            elif remap_remove and instr[0:8] == "//qubits":
                matchobj = re.match(r"//qubits:(\d+)", instr)
                assert matchobj is not None
                ary = matchobj.groups()
                mapping = list(range(int(ary[0])))
            elif instr == "openqasm2.0;" or instr == 'include"qelib1.inc";':
                # related to file format, not for read.
                pass
            else:
                raise RuntimeError(f"unknown line: {instr}")
        return cir
