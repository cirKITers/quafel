from qiskit.circuit.random import random_circuit
from qiskit.compiler import transpile

def log_circuit(qasm_circuit):
    return {
        "circuit_image": None
    }

def generate_random_qasm_circuit(qubits:int, depth:int, measure:bool, seed:int):

    qc = random_circuit(qubits, depth, max_operands=3, measure=measure, seed=seed)
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

    return {
        "qasm_circuit":qc.qasm()
    }


