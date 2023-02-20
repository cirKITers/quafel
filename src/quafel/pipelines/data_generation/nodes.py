from qiskit.circuit.random import random_circuit
from qiskit.compiler import transpile

from typing import List, Dict
import pandas as pd


def log_circuit(qasm_circuit):
    return {"circuit_image": None}


def part_generate_random_qasm_circuit(partition, seed=100):
    qubits = partition[partition.columns[0]][0]
    depth = partition[partition.columns[0]][1]
    return generate_random_qasm_circuit(qubits, depth, seed)


def generate_random_qasm_circuit(qubits: int, depth: int, seed: int):
    qc = random_circuit(qubits, depth, max_operands=3, measure=True, seed=seed)
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


def generate_evaluation_matrix(
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
    qubits = [i for i in range(min_qubits, max_qubits, qubits_increment)]
    depths = [i for i in range(min_depth, max_depth, depth_increment)]
    shots = [i for i in range(min_shots, max_shots, shots_increment)]

    return {
        "evaluation_matrix": {
            "qubits": qubits,
            "depths": depths,
            "shots": shots,
        }
    }


def generate_evaluation_partitions(evaluation_matrix):
    partitions = {}
    idx = 0
    for q in evaluation_matrix["qubits"]:
        for d in evaluation_matrix["depths"]:
            for s in evaluation_matrix["shots"]:
                partitions[f"{idx}"] = {"qubits": q, "depth": d, "shots": s}
                idx += 1

    pass
    return {"evaluation_partitions": pd.DataFrame(partitions)}
