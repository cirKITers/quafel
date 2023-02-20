"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from quafel.pipelines.data_generation.nodes import (
    log_circuit,
    generate_random_qasm_circuit,
    part_generate_random_qasm_circuit,
    generate_evaluation_matrix,
    generate_evaluation_partitions,
)


def create_pipeline(n_partitions=1, **kwargs) -> dict:
    nd_log_circuit = node(
        func=log_circuit,
        inputs={"qasm_circuit": "qasm_circuit"},
        outputs={"circuit_image": "circuit_image"},
    )

    nd_generate_random_qasm_circuit = node(
        func=generate_random_qasm_circuit,
        inputs={
            "qubits": "params:qubits",
            "depth": "params:depth",
            "seed": "params:seed",
        },
        outputs={
            "qasm_circuit": "qasm_circuit",
        },
    )

    nd_generate_evaluation_matrix = node(
        func=generate_evaluation_matrix,
        inputs={
            "min_qubits": "params:min_qubits",
            "max_qubits": "params:max_qubits",
            "qubits_increment": "params:qubits_increment",
            "min_depth": "params:min_depth",
            "max_depth": "params:max_depth",
            "depth_increment": "params:depth_increment",
            "min_shots": "params:min_shots",
            "max_shots": "params:max_shots",
            "max_shots": "params:max_shots",
            "shots_increment": "params:shots_increment",
            "frameworks": "params:frameworks",
        },
        outputs={
            "evaluation_matrix": "evaluation_matrix",
        },
    )

    nd_generate_evaluation_partitions = node(
        func=generate_evaluation_partitions,
        inputs={"evaluation_matrix": "evaluation_matrix"},
        outputs={
            "evaluation_partitions": "evaluation_partitions",
        },
    )

    pl_parallel_generate_and_log_circuit = pipeline(
        [
            nd_generate_evaluation_matrix,
            nd_generate_evaluation_partitions,
            *[
                node(
                    func=part_generate_random_qasm_circuit,
                    inputs={
                        "partition": f"evaluation_partition_{i}",
                        "seed": "params:seed",
                    },
                    outputs={
                        "qasm_circuit": f"qasm_circuit_{i}",
                        "n_shots": f"n_shots_{i}",
                        "framework": f"framework_{i}",
                    },
                    name=f"part_generate_random_qasm_circuit_{i}",
                )
                for i in range(n_partitions)
            ],
        ],
        outputs={
            "evaluation_matrix": "evaluation_matrix",
            "evaluation_partitions": "evaluation_partitions",
            **{
                f"qasm_circuit_{i}": f"qasm_circuit_{i}"
                for i in range(n_partitions)
            },
            **{f"n_shots_{i}": f"n_shots_{i}" for i in range(n_partitions)},
            **{
                f"framework_{i}": f"framework_{i}" for i in range(n_partitions)
            },
        },
        namespace="data_generation",
    )

    pl_generate_and_log_circuit = pipeline(
        [
            nd_generate_random_qasm_circuit,
            # nd_log_circuit
        ],
        outputs={"qasm_circuit": "qasm_circuit"},
        namespace="data_generation",
    )

    pl_generate_evaluation_matrix = pipeline(
        [nd_generate_evaluation_matrix, nd_generate_evaluation_partitions],
        outputs={"evaluation_partitions": "evaluation_partitions"},
        namespace="data_generation",
    )

    return {
        "pl_generate_and_log_circuit": pl_generate_and_log_circuit,
        "pl_parallel_generate_and_log_circuit": pl_parallel_generate_and_log_circuit,
        "pl_generate_evaluation_matrix": pl_generate_evaluation_matrix,
    }
