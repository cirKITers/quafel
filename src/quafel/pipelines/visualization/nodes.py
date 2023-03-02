"""
This is a boilerplate pipeline 'visualization'
generated using Kedro 0.18.4
"""

import plotly.express as px
import plotly.graph_objs as go
from typing import Dict
import pandas as pd


def shots_depths_viz(execution_durations_combined: Dict):
    figures = {}

    grouped_by_fw = execution_durations_combined.groupby("0")

    for fw, qubit_depth_duration in grouped_by_fw:
        figures[fw] = {}

        grouped_by_qubit = qubit_depth_duration.groupby("1")

        for q, depth_duration in grouped_by_qubit:
            # grouped_by_shots_sorted_by_depth = depth_duration.sort_values('2').groupby('3')
            duration_sorted_by_depth = depth_duration.sort_values("2")

            # image = []
            # for s, duration in grouped_by_shots_sorted_by_depth:
            #     image.append(duration['4'].to_numpy())

            figures[fw][q] = go.Figure(
                [
                    go.Heatmap(
                        x=duration_sorted_by_depth["3"],
                        y=duration_sorted_by_depth["2"],
                        z=duration_sorted_by_depth["4"],
                        # colorscale='Viridis'
                    )
                ]
            )
    return {
        "figures": figures,
    }


def shots_qubits_viz(execution_durations_combined: Dict):
    pass
