from qiskit.circuit.random import random_circuit
from qiskit.compiler import transpile

from typing import List, Dict
import pandas as pd


def log_circuit(qasm_circuit):
    return {"circuit_image": None}


def part_generate_random_qasm_circuit(partition, seed=100):
    # TODO: improve this by accessing data by name
    framework = partition[partition.columns[0]][0]
    qubits = int(partition[partition.columns[0]][1])
    depth = int(partition[partition.columns[0]][2])
    shots = int(partition[partition.columns[0]][3])
    return {
        **generate_random_qasm_circuit(qubits, depth, seed),
        "n_shots": shots,
        "framework": framework,
    }


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
    qubits_type: int,
    min_depth: int,
    max_depth: int,
    depth_increment: int,
    depth_type: int,
    min_shots: int,
    max_shots: int,
    shots_increment: int,
    shots_type: int,
    frameworks: List[str],
):
    def generate_ticks(min_t, max_t, inc_t, type_t="linear"):
        if type_t == "linear":
            ticks = [i for i in range(min_t, max_t + inc_t, inc_t)]
        elif "exp" in type_t:
            base = int(type_t.split("exp")[1])
            ticks = [base**i for i in range(min_t, max_t + inc_t, inc_t)]
        else:
            raise ValueError("Unknown base specified and type is not linear")

        return ticks

    qubits = generate_ticks(min_qubits, max_qubits, qubits_increment, qubits_type)
    depths = generate_ticks(min_depth, max_depth, depth_increment, depth_type)
    shots = generate_ticks(min_shots, max_shots, shots_increment, shots_type)

    frameworks = frameworks

    return {
        "evaluation_matrix": {
            "frameworks": frameworks,
            "qubits": qubits,
            "depths": depths,
            "shots": shots,
        }
    }


def generate_evaluation_partitions(evaluation_matrix, skip_combinations):
    partitions = {}
    idx = 0
    for f in evaluation_matrix["frameworks"]:
        if "qubits" in skip_combinations:
            q = max(evaluation_matrix["qubits"])
            for d in evaluation_matrix["depths"]:
                for s in evaluation_matrix["shots"]:
                    partitions[f"{idx}"] = {
                        "framework": f,
                        "qubits": q,
                        "depth": d,
                        "shots": s,
                    }
                    idx += 1
        elif "depth" in skip_combinations:
            d = max(evaluation_matrix["depth"])
            for q in evaluation_matrix["qubtis"]:
                for s in evaluation_matrix["shots"]:
                    partitions[f"{idx}"] = {
                        "framework": f,
                        "qubits": q,
                        "depth": d,
                        "shots": s,
                    }
                    idx += 1
        elif "shots" in skip_combinations:
            s = max(evaluation_matrix["shots"])
            for d in evaluation_matrix["depths"]:
                for q in evaluation_matrix["qubits"]:
                    partitions[f"{idx}"] = {
                        "framework": f,
                        "qubits": q,
                        "depth": d,
                        "shots": s,
                    }
                    idx += 1
        else:
            for q in evaluation_matrix["qubits"]:
                for d in evaluation_matrix["depths"]:
                    for s in evaluation_matrix["shots"]:
                        partitions[f"{idx}"] = {
                            "framework": f,
                            "qubits": q,
                            "depth": d,
                            "shots": s,
                        }
                        idx += 1

    eval_partitions = pd.DataFrame(partitions)

    return {"evaluation_partitions": eval_partitions}
