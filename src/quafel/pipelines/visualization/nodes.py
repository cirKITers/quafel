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

    grouped_by_fw = execution_durations_combined.groupby("framework")

    for fw, qubit_depth_duration in grouped_by_fw:
        grouped_by_qubit = qubit_depth_duration.groupby("qubits")

        for q, depth_duration in grouped_by_qubit:
            # grouped_by_shots_sorted_by_depth = depth_duration.sort_values('2').groupby('3')
            duration_sorted_by_depth = depth_duration.sort_values("depth")
            duration_mean = duration_sorted_by_depth.filter(
                regex=("duration_\d.")
            ).mean(axis=1)
            # image = []
            # for s, duration in grouped_by_shots_sorted_by_depth:
            #     image.append(duration['4'].to_numpy())

            figures[f"{fw}_qubits_{q}"] = go.Figure(
                [
                    go.Heatmap(
                        x=duration_sorted_by_depth["shots"],
                        y=duration_sorted_by_depth["depth"],
                        z=duration_mean,
                        # colorscale='Viridis'
                    )
                ]
            )

        grouped_by_depth = qubit_depth_duration.groupby("depth")

        for d, qubit_duration in grouped_by_depth:
            # grouped_by_shots_sorted_by_depth = depth_duration.sort_values('2').groupby('3')
            duration_sorted_by_qubit = qubit_duration.sort_values("qubits")

            # image = []
            # for s, duration in grouped_by_shots_sorted_by_depth:
            #     image.append(duration['4'].to_numpy())

            figures[f"{fw}_depth_{d}"] = go.Figure(
                [
                    go.Heatmap(
                        x=duration_sorted_by_qubit["shots"],
                        y=duration_sorted_by_qubit["qubits"],
                        z=duration_sorted_by_qubit["mean_duration"],
                        # colorscale='Viridis'
                    )
                ]
            )

    return figures


def shots_qubits_viz(execution_durations_combined: Dict):
    pass
