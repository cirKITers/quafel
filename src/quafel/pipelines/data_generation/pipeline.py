"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from quafel.pipelines.data_generation.nodes import (
    log_circuit,
    extract_partition_data,
    generate_random_qasm_circuit_from_partition,
    calculate_measures,
    generate_evaluation_matrix,
    generate_evaluation_partitions,
)


def create_pipeline(
    partitions, circuit_partitions, extract_partitions, **kwargs
) -> dict:
    # nd_log_circuit = node(
    #     func=log_circuit,
    #     inputs={"qasm_circuit": "qasm_circuit"},
    #     outputs={"circuit_image": "circuit_image"},
    # )

    nd_generate_evaluation_matrix = node(
        func=generate_evaluation_matrix,
        inputs={
            "min_qubits": "params:min_qubits",
            "max_qubits": "params:max_qubits",
            "qubits_increment": "params:qubits_increment",
            "qubits_type": "params:qubits_type",
            "min_depth": "params:min_depth",
            "max_depth": "params:max_depth",
            "depth_increment": "params:depth_increment",
            "depth_type": "params:depth_type",
            "min_shots": "params:min_shots",
            "max_shots": "params:max_shots",
            "max_shots": "params:max_shots",
            "shots_increment": "params:shots_increment",
            "shots_type": "params:shots_type",
            "frameworks": "params:frameworks",
        },
        outputs={
            "evaluation_matrix": "evaluation_matrix",
        },
    )

    nd_generate_evaluation_partitions = node(
        func=generate_evaluation_partitions,
        inputs={
            "evaluation_matrix": "evaluation_matrix",
            "skip_combinations": "params:skip_combinations",
        },
        outputs={
            "evaluation_partitions": "evaluation_partitions",
        },
    )

    pl_generate_evaluation_partitions = pipeline(
        [nd_generate_evaluation_matrix, nd_generate_evaluation_partitions],
        outputs={"evaluation_partitions": "evaluation_partitions"},
        namespace="data_generation",
    )

    pl_generate_qasm_circuits = pipeline(
        [
            # These two nodes are mutually exclusive
            # That means that partitions in 'circuit_partitions' are not contained
            # in 'extract_partitions' which helps to prevent generating
            # existent circuits after resuming a pipeline
            *[
                node(
                    func=generate_random_qasm_circuit_from_partition,
                    inputs={
                        "partition": f"evaluation_partition_{i}",
                        "seed": "params:seed",
                    },
                    outputs={
                        "qasm_circuit": f"qasm_circuit_{i}",
                        "n_shots": f"n_shots_{i}",
                        "framework": f"framework_{i}",
                        "circuit": f"circuit_{i}",
                    },
                    tags=["dynamic"],
                    name=f"generate_random_qasm_circuit_from_partition_{i}",
                )
                for i in circuit_partitions
            ],
            *[
                node(
                    func=extract_partition_data,
                    inputs={
                        "partition": f"evaluation_partition_{i}",
                    },
                    outputs={
                        "qubits": f"qubits_{i}",
                        "depth": f"depth_{i}",
                        "n_shots": f"n_shots_{i}",
                        "framework": f"framework_{i}",
                    },
                    tags=["dynamic"],
                    name=f"part_generate_random_qasm_circuit_{i}",
                )
                for i in extract_partitions
            ],
            *[
                node(
                    func=calculate_measures,
                    inputs={
                        "circuit": f"circuit_{i}",
                        "samples_per_parameter": "params:samples_per_parameter",
                        "haar_samples_per_qubit": "params:haar_samples_per_qubit",
                        "seed": "params:seed",
                    },
                    outputs={
                        "measure": f"measure_{i}",
                    },
                    tags=["dynamic"],
                    name=f"calculate_measures_{i}",
                )
                for i in partitions
            ],
        ],
        inputs={
            # note that this dataset is dynamically created in the hooks, so it is not directly available in the catalog
            **{
                f"evaluation_partition_{i}": f"evaluation_partition_{i}"
                for i in partitions
            },
        },
        outputs={
            **{f"qasm_circuit_{i}": f"qasm_circuit_{i}" for i in circuit_partitions},
            **{f"n_shots_{i}": f"n_shots_{i}" for i in partitions},
            **{f"framework_{i}": f"framework_{i}" for i in partitions},
            **{f"measure_{i}": f"measure_{i}" for i in partitions},
        },
        namespace="data_generation",
    )

    return {
        "pl_generate_evaluation_partitions": pl_generate_evaluation_partitions,
        "pl_generate_qasm_circuits": pl_generate_qasm_circuits,
    }
