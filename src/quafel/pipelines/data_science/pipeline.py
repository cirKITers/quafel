"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from quafel.pipelines.data_science.nodes import (
    measure_execution_durations,
    aggregate_evaluations,
    combine_evaluations,
    aggregate_partitions,
)


def create_pipeline(n_partitions=1, **kwargs) -> dict:
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
                        f"{i}": "params:dummy",
                    },
                    outputs={
                        "execution_duration": f"execution_duration_{i}",
                        "execution_result": f"execution_result_{i}",
                    },
                    tags=["dynamic"],
                    name=f"measure_execution_durations_{i}",
                )
                for i in range(n_partitions)
            ],
        ],
        inputs={
            **{f"qasm_circuit_{i}": f"qasm_circuit_{i}" for i in range(n_partitions)},
            **{f"n_shots_{i}": f"n_shots_{i}" for i in range(n_partitions)},
            **{f"framework_{i}": f"framework_{i}" for i in range(n_partitions)},
        },
        outputs={
            **{
                f"execution_duration_{i}": f"execution_duration_{i}"
                for i in range(n_partitions)
            },
            **{
                f"execution_result_{i}": f"execution_result_{i}"
                for i in range(n_partitions)
            },
        },
        namespace="data_science",
    )

    pl_aggregate_evaluations = pipeline(
        [
            node(
                func=aggregate_evaluations,
                inputs=[f"execution_duration_{i}" for i in range(n_partitions)],
                outputs={
                    "aggregated_evaluations": "execution_durations",
                },
                tags=["static"],
                name=f"aggregate_durations",
            ),
            node(
                func=aggregate_evaluations,
                inputs=[f"execution_result_{i}" for i in range(n_partitions)],
                outputs={
                    "aggregated_evaluations": "execution_results",
                },
                tags=["static"],
                name=f"aggregate_results",
            ),
        ],
        inputs={
            **{
                f"execution_duration_{i}": f"execution_duration_{i}"
                for i in range(n_partitions)
            },
            **{
                f"execution_result_{i}": f"execution_result_{i}"
                for i in range(n_partitions)
            },
        },
        outputs={
            "execution_results": "execution_results",
            "execution_durations": "execution_durations",
        },
        namespace="data_science",
    )

    pl_combine_evaluations = pipeline(
        [
            node(
                func=combine_evaluations,
                inputs={
                    "evaluation_partitions": "evaluation_partitions",
                    "execution_durations": "execution_durations",
                    "execution_results": "execution_results",
                },
                outputs={
                    "evaluations_combined": "evaluations_combined",
                },
                tags=["static"],
                name=f"combine_evaluations",
            ),
        ],
        inputs={
            "evaluation_partitions": "evaluation_partitions",
            "execution_durations": "execution_durations",
            "execution_results": "execution_results",
        },
        outputs={
            "evaluations_combined": "evaluations_combined",
        },
        namespace="data_science",
    )

    return {
        "pl_parallel_measure_execution_durations": pl_parallel_measure_execution_durations
        + pl_aggregate_evaluations
        + pl_combine_evaluations,
    }
