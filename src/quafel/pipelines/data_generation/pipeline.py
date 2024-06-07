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
    extract_circuit,
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
            "min_qubits": "params:data_generation.min_qubits",
            "max_qubits": "params:data_generation.max_qubits",
            "qubits_increment": "params:data_generation.qubits_increment",
            "qubits_type": "params:data_generation.qubits_type",
            "min_depth": "params:data_generation.min_depth",
            "max_depth": "params:data_generation.max_depth",
            "depth_increment": "params:data_generation.depth_increment",
            "depth_type": "params:data_generation.depth_type",
            "min_shots": "params:data_generation.min_shots",
            "max_shots": "params:data_generation.max_shots",
            "max_shots": "params:data_generation.max_shots",
            "shots_increment": "params:data_generation.shots_increment",
            "shots_type": "params:data_generation.shots_type",
            "frameworks": "params:data_generation.frameworks",
        },
        outputs={
            "evaluation_matrix": "evaluation_matrix",
        },
    )

    nd_generate_evaluation_partitions = node(
        func=generate_evaluation_partitions,
        inputs={
            "evaluation_matrix": "evaluation_matrix",
            "skip_combinations": "params:data_generation.skip_combinations",
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
                        "seed": "params:data_generation.seed",
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
                    name=f"extract_partition_data_{i}",
                )
                for i in extract_partitions
            ],
            *[
                node(
                    func=extract_circuit,
                    inputs={
                        "qasm_circuit": f"qasm_circuit_{i}",
                    },
                    outputs={
                        "circuit": f"circuit_{i}",
                    },
                    tags=["dynamic"],
                    name=f"extract_circuit_{i}",
                )
                for i in extract_partitions
            ],
            *[
                node(
                    func=calculate_measures,
                    inputs={
                        "circuit": f"circuit_{i}",
                        "samples_per_parameter": "params:data_generation.samples_per_parameter",
                        "haar_samples_per_qubit": "params:data_generation.haar_samples_per_qubit",
                        "seed": "params:data_generation.seed",
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
    )

    return {
        "pl_generate_evaluation_partitions": pl_generate_evaluation_partitions,
        "pl_generate_qasm_circuits": pl_generate_qasm_circuits,
    }
