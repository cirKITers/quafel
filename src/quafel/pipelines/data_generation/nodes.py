from qiskit.circuit.random import random_circuit
from qiskit.compiler import transpile


def log_circuit(qasm_circuit):
    return {"circuit_image": None}


def generate_random_qasm_circuit(
    qubits: int, depth: int, measure: bool, seed: int
):
    qc = random_circuit(
        qubits, depth, max_operands=3, measure=measure, seed=seed
    )
    qc = transpile(
        qc,
        basis_gates=[
            "u2",
            "u3",
            "cx",
            "id",
            "x",
            "y",
            "z",
            "h",
            "s",
            "t",
            "rx",
            "ry",
            "rz",
            "cx",
            # "cy", #not in qibo framework
            "cz",
            # "ch", #not in qibo framework
            "swap",
            "ccx",
            # "cswap", #not in qibo framework
        ],
    )

    return {"qasm_circuit": qc.qasm()}


def generate_evaluation_matrix_dataset(
    min_qubits: int,
    max_qubits: int,
    qubits_increment: int,
    min_depth: int,
    max_depth: int,
    depth_increment: int,
    min_shots: int,
    max_shots: int,
    shots_increment: int,
):
    qubits = range(min_qubits, max_qubits, qubits_increment)
    depths = range(min_depth, max_depth, depth_increment)
    shots = range(min_shots, max_shots, shots_increment)

    return {
        "evaluation_matrix_dataset": [qubits, depths, shots],
    }
