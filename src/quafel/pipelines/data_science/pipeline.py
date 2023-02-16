"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from quafel.pipelines.data_science.nodes import (
    measure_execution_durations
)


def create_pipeline(**kwargs) -> dict:
    nd_measure_execution_durations = node(
        func=measure_execution_durations,
        inputs={
            "evaluations":"params:evaluations",
            "qasm_circuit":"qasm_circuit",
            "framework_identifier":"params:framework_identifier",
            "n_shots":"params:n_shots",
        },
        outputs={
            "execution_durations":"execution_durations",
            "execution_results":"execution_results",
        }
    )

    pl_measure_execution_durations = pipeline(
        [
            nd_measure_execution_durations
        ],
        inputs={
            "qasm_circuit":"qasm_circuit",
        },
        namespace="data_science"
    )

    return {
        "pl_measure_execution_durations": pl_measure_execution_durations,
    }
