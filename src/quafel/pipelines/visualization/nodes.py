"""
This is a boilerplate pipeline 'visualization'
generated using Kedro 0.18.4
"""

import plotly.graph_objs as go
import plotly.express as px

from typing import Dict
import pandas as pd

from bisect import bisect_left


class design:
    qual_main = px.colors.qualitative.Set1
    qual_second = px.colors.qualitative.Pastel1

    seq_main = px.colors.sequential.Plasma

    title_font_size = 20
    legend_font_size = 18


def rgb_to_rgba(rgb_value, alpha):
    """
    Adds the alpha channel to an RGB Value and returns it as an RGBA Value
    :param rgb_value: Input RGB Value
    :param alpha: Alpha Value to add  in range [0,1]
    :return: RGBA Value
    """
    return f"rgba{rgb_value[3:-1]}, {alpha})"


def get_lcd_time(pd_time):
    si_time = [["ns", "us", "ms", "s"], [1, 3, 6, 9], [1, 1e3, 1e6, 1e9]]
    min_digits = len(str(pd_time.min().min()))
    max_digits = len(str(pd_time.max().max()))

    mid = min_digits + (max_digits - min_digits) // 2
    idx = bisect_left(si_time[1], mid) - 1

    return (si_time[0][idx], si_time[2][idx])


def extract_framework_name_from_id(identifier):
    return identifier.replace("fw", "framework").capitalize().replace("_", " ")


def shots_qubits_viz(evaluations_combined: Dict):
    figures = {}

    grouped_by_fw = evaluations_combined.groupby("framework")

    for fw, qubit_depth_duration in grouped_by_fw:
        framework_name = extract_framework_name_from_id(fw)
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
                        colorscale=design.seq_main,
                    )
                ]
            )
            figures[f"framework_{fw}_qubits_{q}"].update_layout(
                yaxis_title="Circuit Depth",
                xaxis_title="Num. of Shots",
                title=dict(
                    text=f"{framework_name} simulation duration: Circuit Depth and Num. of Shots",
                    font=dict(
                        size=design.title_font_size,
                    ),
                ),
                hovermode="x",
                font=dict(
                    size=design.legend_font_size,
                ),
            )

    return figures


def shots_depths_viz(evaluations_combined: Dict):
    figures = {}

    grouped_by_fw = evaluations_combined.groupby("framework")

    for fw, qubit_depth_duration in grouped_by_fw:
        framework_name = extract_framework_name_from_id(fw)
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
                        colorscale=design.seq_main,
                    )
                ]
            )
            figures[f"framework_{fw}_depth_{d}"].update_layout(
                yaxis_title="Num. of Qubits",
                xaxis_title="Num. of Shots",
                title=dict(
                    text=f"{framework_name} simulation duration: Num. of qubits and Num. of Shots",
                    font=dict(
                        size=design.title_font_size,
                    ),
                ),
                hovermode="x",
                font=dict(
                    size=design.legend_font_size,
                ),
            )

    return figures


def qubits_time_viz(evaluations_combined: Dict):
    figures = {}

    # those two color sets are well suited as they correspond regarding their color value but differ from their luminosity and saturation values
    main_colors_it = iter(design.qual_main)
    sec_colors_it = iter(design.qual_second)

    si_time, factor_time = get_lcd_time(
        evaluations_combined.filter(regex=(r"duration_\d."))
    )

    grouped_by_fw = evaluations_combined.groupby("framework")
    for fw, qubit_depth_duration in grouped_by_fw:
        main_color_sel = next(main_colors_it)
        sec_color_sel = rgb_to_rgba(next(sec_colors_it), 0.2)
        framework_name = extract_framework_name_from_id(fw)

        grouped_by_depth = qubit_depth_duration.groupby("depth")

        for d, qubit_shots_duration in grouped_by_depth:
            grouped_by_shots = qubit_shots_duration.groupby("shots")

            for s, fw_qubit_duration in grouped_by_shots:
                # grouped_by_shots_sorted_by_depth = depth_duration.sort_values('2').groupby('3')
                duration_sorted_by_qubit = fw_qubit_duration.sort_values("qubits")

                durations = duration_sorted_by_qubit.filter(regex=(r"duration_\d."))

                durations_mean = durations.mean(axis=1) / factor_time
                durations_max = durations.max(axis=1) / factor_time
                durations_min = durations.min(axis=1) / factor_time

                # image = []
                # for s, duration in grouped_by_shots_sorted_by_depth:
                #     image.append(duration['4'].to_numpy())

                if f"shots_{s}_depth_{d}" not in figures:
                    figures[f"shots_{s}_depth_{d}"] = go.Figure()

                figures[f"shots_{s}_depth_{d}"].add_trace(
                    go.Scatter(
                        name=f"{framework_name}",
                        x=duration_sorted_by_qubit["qubits"],
                        y=durations_mean,
                        mode="lines",
                        line=dict(color=main_color_sel),
                    )
                )
                figures[f"shots_{s}_depth_{d}"].add_trace(
                    go.Scatter(
                        name=f"{framework_name} - High",
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
                        name=f"{framework_name} - Low",
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
                    yaxis_title=f"Time ({si_time})",
                    xaxis_title="Num. of Qubits",
                    xaxis=dict(tickmode="linear", tick0=1, dtick=1),
                    title=dict(
                        text=f"Framework simulation duration over num. of qubits ({s} shots, circuit depth {d})",
                        font=dict(
                            size=design.title_font_size,
                        ),
                    ),
                    hovermode="x",
                    font=dict(
                        size=design.legend_font_size,
                    ),
                )

    return figures
