"""
This is a boilerplate pipeline 'visualization'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from quafel.pipelines.visualization.nodes import (
    shots_depths_viz,
    shots_qubits_viz,
    qubits_time_viz,
    depth_time_viz,
    shots_time_viz,
)


def create_pipeline(figures, **kwargs) -> dict:
    pl_visualize_evaluations = pipeline(
        [
            node(
                func=shots_depths_viz,
                inputs={
                    "evaluations_combined": "evaluations_combined",
                },
                outputs={
                    **{
                        f: f
                        for f in filter(
                            lambda s: ("framework_" in s) and ("depth_" in s),
                            figures,
                        )
                    },
                },
                tags=["static"],
                name=f"shots_depths_viz",
            ),
            node(
                func=shots_qubits_viz,
                inputs={
                    "evaluations_combined": "evaluations_combined",
                },
                outputs={
                    **{
                        f: f
                        for f in filter(
                            lambda s: ("framework_" in s) and ("qubits_" in s),
                            figures,
                        )
                    },
                },
                tags=["static"],
                name=f"shots_depth_viz",
            ),
            node(
                func=qubits_time_viz,
                inputs={
                    "evaluations_combined": "evaluations_combined",
                },
                outputs={
                    **{
                        f: f
                        for f in filter(
                            lambda s: ("shots_" in s) and ("depth_" in s),
                            figures,
                        )
                    },
                },
                tags=["static"],
                name=f"qubits_time_viz",
            ),
            node(
                func=shots_time_viz,
                inputs={
                    "evaluations_combined": "evaluations_combined",
                },
                outputs={
                    **{
                        f: f
                        for f in filter(
                            lambda s: ("qubits_" in s) and ("depth_" in s),
                            figures,
                        )
                    },
                },
                tags=["static"],
                name=f"shots_time_viz",
            ),
            node(
                func=depth_time_viz,
                inputs={
                    "evaluations_combined": "evaluations_combined",
                },
                outputs={
                    **{
                        f: f
                        for f in filter(
                            lambda s: ("shots_" in s) and ("qubits_" in s),
                            figures,
                        )
                    },
                },
                tags=["static"],
                name=f"depth_time_viz",
            ),
        ],
        inputs={
            "evaluations_combined": "evaluations_combined",
        },
        outputs={
            **{f: f for f in figures},
        },
    )

    return {
        "pl_visualize_evaluations": pl_visualize_evaluations,
    }
