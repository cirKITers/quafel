"""
This is a boilerplate pipeline 'visualization'
generated using Kedro 0.18.4
"""

import plotly.graph_objs as go
import plotly.express as px

from typing import Dict
import pandas as pd


def rgb_to_rgba(rgb_value, alpha):
    """
    Adds the alpha channel to an RGB Value and returns it as an RGBA Value
    :param rgb_value: Input RGB Value
    :param alpha: Alpha Value to add  in range [0,1]
    :return: RGBA Value
    """
    return f"rgba{rgb_value[3:-1]}, {alpha})"


def shots_qubits_viz(evaluations_combined: Dict):
    figures = {}

    grouped_by_fw = evaluations_combined.groupby("framework")

    for fw, qubit_depth_duration in grouped_by_fw:
        framework_name = (
            fw.replace("fw", "framework").capitalize().replace("_", " ")
        )
        grouped_by_qubit = qubit_depth_duration.groupby("qubits")

        for q, depth_duration in grouped_by_qubit:
            # grouped_by_shots_sorted_by_depth = depth_duration.sort_values('2').groupby('3')
            duration_sorted_by_depth = depth_duration.sort_values("depth")
            duration_mean = duration_sorted_by_depth.filter(
                regex=(r"duration_\d.")
            ).mean(axis=1)
            # image = []
            # for s, duration in grouped_by_shots_sorted_by_depth:
            #     image.append(duration['4'].to_numpy())

            figures[f"framework_{fw}_qubits_{q}"] = go.Figure(
                [
                    go.Heatmap(
                        x=duration_sorted_by_depth["shots"],
                        y=duration_sorted_by_depth["depth"],
                        z=duration_mean,
                        # colorscale='Viridis'
                    )
                ]
            )
            figures[f"framework_{fw}_qubits_{q}"].update_layout(
                yaxis_title="Circuit Depth",
                xaxis_title="Num. of Shots",
                title=f"{framework_name} simulation duration: Circuit Depth and Num. of Shots",
                hovermode="x",
            )

    return figures


def shots_depths_viz(evaluations_combined: Dict):
    figures = {}

    grouped_by_fw = evaluations_combined.groupby("framework")

    for fw, qubit_depth_duration in grouped_by_fw:
        framework_name = (
            fw.replace("fw", "framework").capitalize().replace("_", " ")
        )
        grouped_by_depth = qubit_depth_duration.groupby("depth")

        for d, qubit_duration in grouped_by_depth:
            # grouped_by_shots_sorted_by_depth = depth_duration.sort_values('2').groupby('3')
            duration_sorted_by_qubit = qubit_duration.sort_values("qubits")
            duration_mean = duration_sorted_by_qubit.filter(
                regex=(r"duration_\d.")
            ).mean(axis=1)
            # image = []
            # for s, duration in grouped_by_shots_sorted_by_depth:
            #     image.append(duration['4'].to_numpy())

            figures[f"framework_{fw}_depth_{d}"] = go.Figure(
                [
                    go.Heatmap(
                        x=duration_sorted_by_qubit["shots"],
                        y=duration_sorted_by_qubit["qubits"],
                        z=duration_mean,
                        # colorscale='Viridis'
                    )
                ]
            )
            figures[f"framework_{fw}_depth_{d}"].update_layout(
                yaxis_title="Num. of Qubits",
                xaxis_title="Num. of Shots",
                title=f"{framework_name} simulation duration: Num. of qubits and Num. of Shots",
                hovermode="x",
            )

    return figures


def qubits_time_viz(evaluations_combined: Dict):
    figures = {}

    # those two color sets are well suited as they correspond regarding their color value but differ from their luminosity and saturation values
    main_colors_it = iter(px.colors.qualitative.Set2)
    sec_colors_it = iter(px.colors.qualitative.Pastel2)

    grouped_by_fw = evaluations_combined.groupby("framework")
    for fw, qubit_depth_duration in grouped_by_fw:
        main_color_sel = next(main_colors_it)
        sec_color_sel = rgb_to_rgba(next(sec_colors_it), 0.2)

        grouped_by_depth = qubit_depth_duration.groupby("depth")

        for d, qubit_shots_duration in grouped_by_depth:
            grouped_by_shots = qubit_shots_duration.groupby("shots")

            for s, fw_qubit_duration in grouped_by_shots:
                # grouped_by_shots_sorted_by_depth = depth_duration.sort_values('2').groupby('3')
                duration_sorted_by_qubit = fw_qubit_duration.sort_values(
                    "qubits"
                )

                durations = duration_sorted_by_qubit.filter(
                    regex=(r"duration_\d.")
                )

                durations_mean = durations.mean(axis=1)
                durations_max = durations.max(axis=1)
                durations_min = durations.min(axis=1)

                # image = []
                # for s, duration in grouped_by_shots_sorted_by_depth:
                #     image.append(duration['4'].to_numpy())

                traces = []

                if f"shots_{s}_depth_{d}" not in figures:
                    figures[f"shots_{s}_depth_{d}"] = go.Figure()

                figures[f"shots_{s}_depth_{d}"].add_trace(
                    go.Scatter(
                        name=f"{fw}",
                        x=duration_sorted_by_qubit["qubits"],
                        y=durations_mean,
                        mode="lines",
                        line=dict(color=main_color_sel),
                    )
                )
                figures[f"shots_{s}_depth_{d}"].add_trace(
                    go.Scatter(
                        name=f"{fw} - High",
                        x=duration_sorted_by_qubit["qubits"],
                        y=durations_max,
                        mode="lines",
                        marker=dict(color="#444"),
                        line=dict(width=0),
                        showlegend=False,
                    )
                )
                figures[f"shots_{s}_depth_{d}"].add_trace(
                    go.Scatter(
                        name=f"{fw} - Low",
                        x=duration_sorted_by_qubit["qubits"],
                        y=durations_min,
                        marker=dict(color="#444"),
                        line=dict(width=0),
                        mode="lines",
                        fillcolor=sec_color_sel,
                        fill="tonexty",
                        showlegend=False,
                    )
                )
                figures[f"shots_{s}_depth_{d}"].update_layout(
                    yaxis_title="Time (ns)",
                    xaxis_title="Num. of Qubits",
                    title="Framework simulation duration over num. of qubits",
                    hovermode="x",
                )

    return figures
