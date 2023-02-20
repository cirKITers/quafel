"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from quafel.pipelines.data_science.nodes import (
    measure_execution_durations,
    aggregate_evaluations,
)

import pandas as pd


def create_pipeline(n_partitions=1, **kwargs) -> dict:
    nd_measure_execution_durations = node(
        func=measure_execution_durations,
        inputs={
            "evaluations": "params:evaluations",
            "qasm_circuit": "qasm_circuit",
            "framework_identifier": "params:framework_identifier",
            "n_shots": "params:n_shots",
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
                        "framework_identifier": "params:framework_identifier",
                        "n_shots": "params:n_shots",
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
                    "execution_results": "execution_results",
                },
                name=f"aggregate_evaluations",
            ),
        ],
        inputs=[f"qasm_circuit_{i}" for i in range(n_partitions)],
        outputs={
            "execution_durations": "execution_duration_partitions",
            "execution_results": "execution_result_partitions",
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
