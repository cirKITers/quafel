"""
This is a boilerplate pipeline 'visualization'
generated using Kedro 0.18.4
"""

import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio

from typing import Dict, List
import pandas as pd

from bisect import bisect_left
from math import log10, floor

import numpy as np

import os

duration_regex = r"duration_\d*"


class design:
    qual_main = px.colors.qualitative.Dark2  # set1
    qual_second = px.colors.qualitative.Pastel2  # pastel1

    seq_main = px.colors.sequential.Viridis  # pastel1

    print_figure_title = False

    title_font_size = 18
    legend_font_size = 16

    legend_x_pos = 0
    legend_y_pos = 1

    scatter_legend = dict(
        x=legend_x_pos,
        y=legend_y_pos,
        orientation="h",
        traceorder="normal",
        font=dict(
            size=legend_font_size,
        ),
    )
    base_theme = "simple_white"

    include_framework_term = False


def rgb_to_rgba(rgb_value, alpha):
    """
    Adds the alpha channel to an RGB Value and returns it as an RGBA Value
    :param rgb_value: Input RGB Value
    :param alpha: Alpha Value to add  in range [0,1]
    :return: RGBA Value
    """
    return f"rgba{rgb_value[3:-1]}, {alpha})"


def get_time_scale(pd_time):
    n_evals = pd_time.shape[1]

    def find_exp(number) -> int:
        """
        From https://stackoverflow.com/questions/64183806/extracting-the-exponent-from-scientific-notation
        """
        base10 = log10(abs(number))
        return floor(base10)

    mid = np.mean(
        [
            find_exp(pd_time.min().min() / n_evals),
            find_exp(pd_time.max().max() / n_evals),
        ]
    ).astype(int)

    si_time = [["ns", "us", "ms", "s"], [-9, -6, -3, 0], [1e9, 1e6, 1e3, 1]]

    idx = bisect_left(si_time[1], mid)

    return (si_time[0][idx], si_time[2][idx] / n_evals)


def extract_framework_name_from_id(identifier):
    """
    Global method to generate display name from class name.
    E.g.: qiskit_fw -> Qiskit Framework
    """
    if design.include_framework_term:
        return identifier.replace("fw", "framework").capitalize().replace("_", " ")
    else:
        return identifier.replace("fw", "").capitalize().replace("_", " ")


def shots_qubits_viz(evaluations_combined: Dict):
    figures = {}

    si_time, factor_time = get_time_scale(
        evaluations_combined.filter(regex=duration_regex)
    )

    grouped_by_fw = evaluations_combined.groupby("framework")

    for fw, qubit_depth_duration in grouped_by_fw:
        framework_name = extract_framework_name_from_id(fw)
        grouped_by_qubit = qubit_depth_duration.groupby("qubits")

        for q, depth_duration in grouped_by_qubit:
            duration_sorted_by_depth = depth_duration.sort_values("depth")
            durations = duration_sorted_by_depth.filter(regex=duration_regex)

            durations *= factor_time

            duration_mean = durations.mean(axis=1)

            # Divide by the number of evals
            duration_mean /= len(duration_sorted_by_depth.filter(regex=duration_regex))

            figures[f"framework_{fw}_qubits_{q}"] = go.Figure(
                [
                    go.Heatmap(
                        x=duration_sorted_by_depth["shots"],
                        y=duration_sorted_by_depth["depth"],
                        z=duration_mean,
                        colorscale=design.seq_main,
                        colorbar=dict(title=f"Time ({si_time})"),
                    )
                ]
            )
            figures[f"framework_{fw}_qubits_{q}"].update_layout(
                yaxis_title="Circuit Depth",
                xaxis_title="Num. of Shots",
                title=dict(
                    text=f"{framework_name} simulation duration: Circuit Depth and Num. of Shots"
                    if design.print_figure_title
                    else "",
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

    si_time, factor_time = get_time_scale(
        evaluations_combined.filter(regex=duration_regex)
    )

    grouped_by_fw = evaluations_combined.groupby("framework")

    for fw, qubit_depth_duration in grouped_by_fw:
        framework_name = extract_framework_name_from_id(fw)
        grouped_by_depth = qubit_depth_duration.groupby("depth")

        for d, qubit_duration in grouped_by_depth:
            # grouped_by_shots_sorted_by_depth = depth_duration.sort_values('2').groupby('3')
            duration_sorted_by_qubit = qubit_duration.sort_values("qubits")
            durations = duration_sorted_by_qubit.filter(regex=duration_regex)

            durations *= factor_time

            duration_mean = durations.mean(axis=1)
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
                        colorbar=dict(title=f"Time ({si_time})"),
                    )
                ]
            )
            figures[f"framework_{fw}_depth_{d}"].update_layout(
                yaxis_title="Num. of Qubits",
                xaxis_title="Num. of Shots",
                title=dict(
                    text=f"{framework_name} simulation duration: Num. of qubits and Num. of Shots"
                    if design.print_figure_title
                    else "",
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


def qubits_time_viz(evaluations_combined: Dict, skip_frameworks: List):
    figures = {}

    # those two color sets are well suited as they correspond regarding their color value but differ from their luminosity and saturation values
    main_colors_it = iter(design.qual_main)
    sec_colors_it = iter(design.qual_second)

    si_time, factor_time = get_time_scale(
        evaluations_combined.filter(regex=duration_regex)
    )

    grouped_by_fw = evaluations_combined.groupby("framework")
    for fw, qubit_depth_duration in grouped_by_fw:
        if fw in skip_frameworks:
            continue

        main_color_sel = next(main_colors_it)
        sec_color_sel = rgb_to_rgba(next(sec_colors_it), 0.2)
        framework_name = extract_framework_name_from_id(fw)

        grouped_by_depth = qubit_depth_duration.groupby("depth")

        for d, qubit_shots_duration in grouped_by_depth:
            grouped_by_shots = qubit_shots_duration.groupby("shots")

            for s, fw_qubit_duration in grouped_by_shots:
                # grouped_by_shots_sorted_by_depth = depth_duration.sort_values('2').groupby('3')
                duration_sorted_by_qubit = fw_qubit_duration.sort_values("qubits")

                durations = duration_sorted_by_qubit.filter(regex=duration_regex)
                durations *= factor_time

                durations_mean = durations.mean(axis=1)
                durations_max = durations.max(axis=1)
                durations_min = durations.min(axis=1)

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
                    xaxis=dict(
                        title="Num. of Qubits",
                        tickmode="linear",
                        tick0=1,
                        dtick=1,
                        showgrid=False,
                    ),
                    yaxis=dict(title=f"Time ({si_time})"),
                    title=dict(
                        text=f"Framework simulation duration over num. of qubits ({s} shots, circuit depth {d})"
                        if design.print_figure_title
                        else "",
                        font=dict(
                            size=design.title_font_size,
                        ),
                    ),
                    hovermode="x",
                    font=dict(
                        size=design.legend_font_size,
                    ),
                    legend=design.scatter_legend,
                    template=design.base_theme,
                )

    return figures


def shots_time_viz(evaluations_combined: Dict, skip_frameworks: List):
    figures = {}

    # those two color sets are well suited as they correspond regarding their color value but differ from their luminosity and saturation values
    main_colors_it = iter(design.qual_main)
    sec_colors_it = iter(design.qual_second)

    si_time, factor_time = get_time_scale(
        evaluations_combined.filter(regex=duration_regex)
    )

    grouped_by_fw = evaluations_combined.groupby("framework")
    for fw, qubit_depth_duration in grouped_by_fw:
        if fw in skip_frameworks:
            continue

        main_color_sel = next(main_colors_it)
        sec_color_sel = rgb_to_rgba(next(sec_colors_it), 0.2)
        framework_name = extract_framework_name_from_id(fw)

        grouped_by_depth = qubit_depth_duration.groupby("depth")

        for d, qubit_shots_duration in grouped_by_depth:
            grouped_by_qubits = qubit_shots_duration.groupby("qubits")

            for q, fw_shots_duration in grouped_by_qubits:
                # grouped_by_shots_sorted_by_depth = depth_duration.sort_values('2').groupby('3')
                duration_sorted_by_shots = fw_shots_duration.sort_values("shots")

                durations = duration_sorted_by_shots.filter(regex=duration_regex)
                durations *= factor_time

                durations_mean = durations.mean(axis=1)
                durations_max = durations.max(axis=1)
                durations_min = durations.min(axis=1)

                # image = []
                # for s, duration in grouped_by_shots_sorted_by_depth:
                #     image.append(duration['4'].to_numpy())

                if f"qubits_{q}_depth_{d}" not in figures:
                    figures[f"qubits_{q}_depth_{d}"] = go.Figure()

                figures[f"qubits_{q}_depth_{d}"].add_trace(
                    go.Scatter(
                        name=f"{framework_name}",
                        x=duration_sorted_by_shots["shots"],
                        y=durations_mean,
                        mode="lines",
                        line=dict(color=main_color_sel),
                    )
                )
                figures[f"qubits_{q}_depth_{d}"].add_trace(
                    go.Scatter(
                        name=f"{framework_name} - High",
                        x=duration_sorted_by_shots["shots"],
                        y=durations_max,
                        mode="lines",
                        marker=dict(color="#444"),
                        line=dict(width=0),
                        showlegend=False,
                    )
                )
                figures[f"qubits_{q}_depth_{d}"].add_trace(
                    go.Scatter(
                        name=f"{framework_name} - Low",
                        x=duration_sorted_by_shots["shots"],
                        y=durations_min,
                        marker=dict(color="#444"),
                        line=dict(width=0),
                        mode="lines",
                        fillcolor=sec_color_sel,
                        fill="tonexty",
                        showlegend=False,
                    )
                )
                figures[f"qubits_{q}_depth_{d}"].update_layout(
                    xaxis=dict(
                        tickmode="linear",
                        tick0=600,
                        dtick=500,
                        title="Num. of Shots",
                        showgrid=False,
                    ),
                    yaxis=dict(title=f"Time ({si_time})"),
                    title=dict(
                        text=f"Framework simulation duration over num. of shots ({q} qubits, circuit depth {d})"
                        if design.print_figure_title
                        else "",
                        font=dict(
                            size=design.title_font_size,
                        ),
                    ),
                    hovermode="x",
                    font=dict(
                        size=design.legend_font_size,
                    ),
                    legend=design.scatter_legend,
                    template=design.base_theme,
                )

    return figures


def depth_time_viz(evaluations_combined: Dict, skip_frameworks: List):
    figures = {}

    # those two color sets are well suited as they correspond regarding their color value but differ from their luminosity and saturation values
    main_colors_it = iter(design.qual_main)
    sec_colors_it = iter(design.qual_second)

    si_time, factor_time = get_time_scale(
        evaluations_combined.filter(regex=duration_regex)
    )

    grouped_by_fw = evaluations_combined.groupby("framework")
    for fw, qubit_depth_duration in grouped_by_fw:
        if fw in skip_frameworks:
            continue

        main_color_sel = next(main_colors_it)
        sec_color_sel = rgb_to_rgba(next(sec_colors_it), 0.2)
        framework_name = extract_framework_name_from_id(fw)

        grouped_by_qubits = qubit_depth_duration.groupby("qubits")

        for q, depth_shots_duration in grouped_by_qubits:
            grouped_by_shots = depth_shots_duration.groupby("shots")

            for s, fw_depth_duration in grouped_by_shots:
                # grouped_by_shots_sorted_by_depth = depth_duration.sort_values('2').groupby('3')
                duration_sorted_by_depth = fw_depth_duration.sort_values("depth")

                durations = duration_sorted_by_depth.filter(regex=duration_regex)
                durations *= factor_time

                durations_mean = durations.mean(axis=1)
                durations_max = durations.max(axis=1)
                durations_min = durations.min(axis=1)

                # image = []
                # for s, duration in grouped_by_shots_sorted_by_depth:
                #     image.append(duration['4'].to_numpy())

                if f"shots_{s}_qubits_{q}" not in figures:
                    figures[f"shots_{s}_qubits_{q}"] = go.Figure()

                figures[f"shots_{s}_qubits_{q}"].add_trace(
                    go.Scatter(
                        name=f"{framework_name}",
                        x=duration_sorted_by_depth["depth"],
                        y=durations_mean,
                        mode="lines",
                        line=dict(color=main_color_sel),
                    )
                )
                figures[f"shots_{s}_qubits_{q}"].add_trace(
                    go.Scatter(
                        name=f"{framework_name} - High",
                        x=duration_sorted_by_depth["depth"],
                        y=durations_max,
                        mode="lines",
                        marker=dict(color="#444"),
                        line=dict(width=0),
                        showlegend=False,
                    )
                )
                figures[f"shots_{s}_qubits_{q}"].add_trace(
                    go.Scatter(
                        name=f"{framework_name} - Low",
                        x=duration_sorted_by_depth["depth"],
                        y=durations_min,
                        marker=dict(color="#444"),
                        line=dict(width=0),
                        mode="lines",
                        fillcolor=sec_color_sel,
                        fill="tonexty",
                        showlegend=False,
                    )
                )
                figures[f"shots_{s}_qubits_{q}"].update_layout(
                    xaxis=dict(
                        tickmode="linear",
                        tick0=1,
                        dtick=10,
                        title="Circuit Depth",
                        showgrid=False,
                    ),
                    yaxis=dict(
                        title=f"Time ({si_time})",
                        showgrid=False,
                    ),
                    title=dict(
                        text=f"Framework simulation duration over circuit depth ({s} shots, {q} qubits)"
                        if design.print_figure_title
                        else "",
                        font=dict(
                            size=design.title_font_size,
                        ),
                    ),
                    hovermode="x",
                    font=dict(
                        size=design.legend_font_size,
                    ),
                    legend=design.scatter_legend,
                    template=design.base_theme,
                )

    return figures


def export_selected(selected_figures, output_folder, **figure):
    for name, fig in figure.items():
        if name in selected_figures:
            pio.full_figure_for_development(
                fig, warn=False
            )  # Disable warnings to prevent printing a box at the bottom left of the figure. See this issue: https://github.com/plotly/plotly.py/issues/3469

            fig.write_image(
                os.path.join(output_folder, f"{name}.pdf"), engine="kaleido"
            )

    return {}
