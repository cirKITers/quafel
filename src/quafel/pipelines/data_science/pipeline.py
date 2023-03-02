"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from quafel.pipelines.data_science.nodes import (
    measure_execution_durations,
    aggregate_evaluations,
    combine_execution_durations,
)


def create_pipeline(n_partitions=1, **kwargs) -> dict:
    nd_measure_execution_durations = node(
        func=measure_execution_durations,
        inputs={
            "evaluations": "params:evaluations",
            "qasm_circuit": "qasm_circuit",
            "n_shots": "params:n_shots",
            "framework_id": "params:framework_identifier",
        },
        outputs={
            "execution_duration": "execution_duration",
            "execution_result": "execution_result",
        },
    )

    pl_parallel_measure_execution_durations = pipeline(
        [
            *[
                node(
                    func=measure_execution_durations,
                    inputs={
                        "evaluations": "params:evaluations",
                        "qasm_circuit": f"qasm_circuit_{i}",
                        "n_shots": f"n_shots_{i}",
                        "framework_id": f"framework_{i}",
                    },
                    outputs={
                        "execution_duration": f"execution_duration_{i}",
                        "execution_result": f"execution_result_{i}",
                    },
                    name=f"measure_execution_durations_{i}",
                )
                for i in range(n_partitions)
            ],
            node(
                func=aggregate_evaluations,
                inputs=[
                    *[f"execution_duration_{i}" for i in range(n_partitions)],
                    *[f"execution_result_{i}" for i in range(n_partitions)],
                ],
                outputs={
                    "execution_durations": "execution_durations",
                    # "execution_results": "execution_results",
                },
                name=f"aggregate_evaluations",
            ),
            node(
                func=combine_execution_durations,
                inputs={
                    "evaluation_partitions": "evaluation_partitions",
                    "execution_durations": "execution_durations",
                },
                outputs={
                    "execution_durations_combined": "execution_durations_combined",
                },
            ),
        ],
        inputs={
            **{
                f"qasm_circuit_{i}": f"qasm_circuit_{i}"
                for i in range(n_partitions)
            },
            **{f"n_shots_{i}": f"n_shots_{i}" for i in range(n_partitions)},
            **{
                f"framework_{i}": f"framework_{i}" for i in range(n_partitions)
            },
            "evaluation_partitions": "evaluation_partitions",
        },
        outputs={
            "execution_durations": "execution_duration_partitions",
            # "execution_results": "execution_result_partitions",
            "execution_durations_combined": "execution_durations_combined",
        },
        namespace="data_science",
    )

    pl_measure_execution_durations = pipeline(
        [nd_measure_execution_durations],
        inputs={
            "qasm_circuit": "qasm_circuit",
        },
        namespace="data_science",
    )

    return {
        "pl_measure_execution_durations": pl_measure_execution_durations,
        "pl_parallel_measure_execution_durations": pl_parallel_measure_execution_durations,
    }
