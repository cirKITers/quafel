"""
This is a boilerplate pipeline 'visualization'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from quafel.pipelines.visualization.nodes import shots_depths_viz


def create_pipeline(figures, **kwargs) -> dict:
    nd_shots_depths_viz = node(
        func=shots_depths_viz,
        inputs={
            "execution_durations_combined": "execution_durations_combined",
        },
        outputs={"figures_shots_depths": "figures_shots_depths"},
    )

    nd_shots_qubits_viz = node(
        func=shots_depths_viz,
        inputs={
            "execution_durations_combined": "execution_durations_combined",
        },
        outputs={"circuit_image": "circuit_image"},
    )

    pl_visualize_evaluations = pipeline(
        [
            node(
                func=shots_depths_viz,
                inputs={
                    "execution_durations_combined": "execution_durations_combined",
                },
                outputs={
                    **{f: f for f in figures},
                },
                name=f"shots_depths_viz",
            )
        ],
        inputs={
            "execution_durations_combined": "execution_durations_combined",
        },
        outputs={
            **{f: f for f in figures},
        },
    )

    return {
        "pl_visualize_evaluations": pl_visualize_evaluations,
    }
