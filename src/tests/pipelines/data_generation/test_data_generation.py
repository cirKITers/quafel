from quafel.pipelines.data_generation.nodes import (
    generate_random_qasm_circuit,
    generate_evaluation_matrix,
    qasm_circuit_to_qiskit,
    calculate_measures,
)


def test_circuit_generation():
    generate_random_qasm_circuit(5, 5, 0)


def test_evaluation_matrix():
    generate_evaluation_matrix(
        5, 5, 5, "linear", 5, 5, 5, "linear", 5, 5, 5, "linear", ["qiskit"]
    )


def test_circuit_conversion():
    qasm_circuit = generate_random_qasm_circuit(5, 5, 0)["qasm_circuit"]
    qasm_circuit_to_qiskit(qasm_circuit)["qiskit_circuit"]


def test_calculate_measures():
    qiskit_circuit = generate_random_qasm_circuit(5, 5, 0)["qiskit_circuit"]
    calculate_measures(
        circuit=qiskit_circuit,
        samples_per_parameter=5,
        haar_samples_per_qubit=5,
        seed=1000,
    )
