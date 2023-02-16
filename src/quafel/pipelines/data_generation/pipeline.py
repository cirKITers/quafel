"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from quafel.pipelines.data_generation.nodes import (
    log_circuit,
    generate_random_qasm_circuit
)


def create_pipeline(**kwargs) -> dict:
    nd_log_circuit = node(
        func=log_circuit,
        inputs={
            "qasm_circuit":"qasm_circuit"
        },
        outputs=
        {
            "circuit_image":"circuit_image"
        }
    )

    nd_generate_random_qasm_circuit = node(
        func=generate_random_qasm_circuit,
        inputs={
            "qubits":"params:qubits",
            "depth":"params:depth",
            "measure":"params:measure",
            "seed":"params:seed",
        },
        outputs={
            "qasm_circuit":"qasm_circuit",
        }
    )

    pl_generate_and_log_circuit = pipeline(
        [
            nd_generate_random_qasm_circuit,
            # nd_log_circuit
        ],
        outputs={
            "qasm_circuit":"qasm_circuit"
        },
        namespace="data_generation"
    )


    return {
        "pl_generate_and_log_circuit": pl_generate_and_log_circuit,
    }
