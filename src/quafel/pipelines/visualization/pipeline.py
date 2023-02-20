"""
This is a boilerplate pipeline 'visualization'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from quafel.pipelines.visualization.nodes import shots_depths_viz


def create_pipeline(**kwargs) -> dict:
    nd_shots_depths_viz = node(
        func=shots_depths_viz,
        inputs={
            "evaluation_partitions": "evaluation_partitions",
            "execution_durations": "execution_duration_partitions",
        },
        outputs={"plotly_shots_depth": "plotly_shots_depth"},
    )

    nd_shots_qubits_viz = node(
        func=shots_depths_viz,
        inputs={
            "evaluation_partitions": "evaluation_partitions",
            "execution_durations": "execution_duration_partitions",
        },
        outputs={"circuit_image": "circuit_image"},
    )

    pl_visualize_evaluations = pipeline(
        [nd_shots_depths_viz, nd_shots_qubits_viz],
        inputs={
            "evaluation_partitions": "evaluation_partitions",
            "execution_durations": "execution_duration_partitions",
        },
    )

    return {
        "pl_visualize_evaluations": pl_visualize_evaluations,
    }
