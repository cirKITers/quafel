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
import logging


import numpy as np

import os

duration_perf_regex = r"duration_perf_\d*"
duration_proc_regex = r"duration_proc_\d*"


class design:
    # see https://plotly.com/python/discrete-color/
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

    qubits_tickangle = 0
    depth_tickangle = 0
    shots_tickangle = -40
    showgrid = False

    time_tick_type = "log"
    time_dtick = 1
    qubits_tick_type = "linear"
    shots_tick_type = "log"
    depth_tick_type = "log"
    depth_tick_mode = "array"  # array
    depth_tick0 = None
    depth_dtick = None  # np.log10(2)


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
            find_exp(pd_time.min().min()),
            find_exp(pd_time.max().max()),
        ]
    ).astype(int)

    si_time = [["ns", "us", "ms", "s"], [-9, -6, -3, 0], [1e9, 1e6, 1e3, 1]]

    idx = bisect_left(si_time[1], mid)

    return (si_time[0][idx], si_time[2][idx])


def extract_framework_name_from_id(identifier):
    """
    Global method to generate display name from class name.
    E.g.: qiskit_fw -> Qiskit Framework
    """
    if design.include_framework_term:
        return identifier.replace("fw", "framework").capitalize().replace("_", " ")
    else:
        return identifier.replace("fw", "").capitalize().replace("_", " ")


def heatmap_viz(x, y, z, z_title, x_title, y_title, plot_title):
    fig = go.Figure(
        [
            go.Heatmap(
                x=x,
                y=y,
                z=z,
                colorscale=design.seq_main_log(len(z)),
                colorbar=dict(
                    title=z_title,
                    tick0=design.tickvals_0,
                    tickmode="array",
                    tickvals=design.tickvals_log(len(z)),
                ),
            )
        ]
    )
    fig.update_layout(
        yaxis=dict(
            type=design.depth_tick_type,
            tickmode=design.depth_tick_mode,
            tickvals=y if design.depth_tick_mode == "array" else None,
            # ticktext=[
            #     f"2^{i}"
            #     for i in duration_sorted_by_shots["qubits"].astype(int)
            # ],
            tick0=design.depth_tick0,
            dtick=design.depth_dtick,
            title=y_title,
            showgrid=design.showgrid,
        ),
        xaxis=dict(
            type=design.shots_tick_type,
            tickmode="array",
            tickvals=x,
            # ticktext=[
            #     f"2^{i}"
            #     for i in duration_sorted_by_shots["qubits"].astype(int)
            # ],
            tickangle=design.shots_tickangle,
            title=x_title,
            showgrid=design.showgrid,
        ),
        title=dict(
            text=plot_title if design.print_figure_title else "",
            font=dict(
                size=design.title_font_size,
            ),
        ),
        hovermode="x",
        font=dict(
            size=design.legend_font_size,
        ),
    )
    return fig


def shots_qubits_viz(evaluations_combined: Dict):
    figures = {}

    si_time, factor_time = get_time_scale(
        evaluations_combined.filter(regex=duration_perf_regex)
    )

    grouped_by_fw = evaluations_combined.groupby("framework")

    for fw, qubit_depth_duration in grouped_by_fw:
        framework_name = extract_framework_name_from_id(fw)
        grouped_by_qubit = qubit_depth_duration.groupby("qubits")

        for q, depth_duration in grouped_by_qubit:
            duration_sorted_by_depth = depth_duration.sort_values("depth")
            durations = duration_sorted_by_depth.filter(regex=duration_perf_regex)

            durations *= factor_time

            duration_mean = durations.mean(axis=1)

            q = int(q)

            figures[f"{fw}_qubits_{q}"] = heatmap_viz(
                x=duration_sorted_by_depth["shots"].astype(int),
                y=duration_sorted_by_depth["depth"].astype(int),
                z=duration_mean,
                z_title=f"Time ({si_time})",
                x_title="Num. of Shots",
                y_title="Circuit Depth",
                plot_title=f"{framework_name} @ {q} Qubits: Circuit Depth and Num. of Shots",
            )

    return figures


def shots_depths_viz(evaluations_combined: Dict):
    figures = {}

    si_time, factor_time = get_time_scale(
        evaluations_combined.filter(regex=duration_perf_regex)
    )

    grouped_by_fw = evaluations_combined.groupby("framework")

    for fw, qubit_depth_duration in grouped_by_fw:
        framework_name = extract_framework_name_from_id(fw)
        grouped_by_depth = qubit_depth_duration.groupby("depth")

        for d, qubit_duration in grouped_by_depth:
            # grouped_by_shots_sorted_by_depth = depth_duration.sort_values('2').groupby('3')
            duration_sorted_by_qubit = qubit_duration.sort_values("qubits")
            durations = duration_sorted_by_qubit.filter(regex=duration_perf_regex)

            durations *= factor_time

            duration_mean = durations.mean(axis=1)

            d = int(d)
            # image = []
            # for s, duration in grouped_by_shots_sorted_by_depth:
            #     image.append(duration['4'].to_numpy())

            figures[f"{fw}_depth_{d}"] = heatmap_viz(
                x=duration_sorted_by_qubit["shots"].astype(int),
                y=duration_sorted_by_qubit["qubits"].astype(int),
                z=duration_mean,
                z_title=f"Time ({si_time})",
                x_title="Num. of Shots",
                y_title="Num. of Qubits",
                plot_title=f"{framework_name} @ Circuit Depth {d}: Num. of qubits and num. of Shots",
            )

    return figures


def depth_qubits_viz(evaluations_combined: Dict):
    figures = {}

    si_time, factor_time = get_time_scale(
        evaluations_combined.filter(regex=duration_perf_regex)
    )

    grouped_by_fw = evaluations_combined.groupby("framework")

    for fw, qubit_depth_duration in grouped_by_fw:
        framework_name = extract_framework_name_from_id(fw)
        grouped_by_shots = qubit_depth_duration.groupby("shots")

        for s, shots_duration in grouped_by_shots:
            duration_sorted_by_depth = shots_duration.sort_values("depth")
            durations = duration_sorted_by_depth.filter(regex=duration_perf_regex)

            durations *= factor_time

            duration_mean = durations.mean(axis=1)

            s = int(s)

            figures[f"{fw}_shots_{s}"] = heatmap_viz(
                x=duration_sorted_by_depth["qubits"].astype(int),
                y=duration_sorted_by_depth["depth"].astype(int),
                z=duration_mean,
                z_title=f"Time ({si_time})",
                x_title="Num. of Shots",
                y_title="Num. of Qubits",
                plot_title=f"{framework_name} @ {s} Shots: Circuit Depth and num. of Qubits",
            )

    return figures


def qubits_time_viz(evaluations_combined: Dict, skip_frameworks: List):
    figures = {}

    # those two color sets are well suited as they correspond regarding their color value but differ from their luminosity and saturation values
    main_colors_it = iter(design.qual_main)
    sec_colors_it = iter(design.qual_second)

    si_time, factor_time = get_time_scale(
        evaluations_combined.filter(regex=duration_perf_regex)
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

                durations = duration_sorted_by_qubit.filter(regex=duration_perf_regex)
                durations *= factor_time

                durations_mean = durations.mean(axis=1)
                durations_max = durations.max(axis=1)
                durations_min = durations.min(axis=1)

                d = int(d)
                s = int(s)
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
                        x=duration_sorted_by_qubit["qubits"].astype(int),
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
                        x=duration_sorted_by_qubit["qubits"].astype(int),
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
                        type=design.qubits_tick_type,
                        tickmode="array",
                        tickvals=duration_sorted_by_qubit["qubits"].astype(int),
                        # ticktext=[
                        #     f"2^{i}"
                        #     for i in duration_sorted_by_shots["qubits"].astype(int)
                        # ],
                        tickangle=design.qubits_tickangle,
                        title="Num. of Qubits",
                        showgrid=design.showgrid,
                    ),
                    yaxis=dict(
                        title=f"Time ({si_time})",
                        type=design.time_tick_type,
                        dtick=design.time_dtick,
                        showgrid=design.showgrid,
                    ),
                    title=dict(
                        text=f"Duration per Framework over num. of Qubits @ {s} Shots, Circuit Depth {d}"
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
        evaluations_combined.filter(regex=duration_perf_regex)
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

                durations = duration_sorted_by_shots.filter(regex=duration_perf_regex)
                durations *= factor_time

                durations_mean = durations.mean(axis=1)
                durations_max = durations.max(axis=1)
                durations_min = durations.min(axis=1)

                q = int(q)
                d = int(d)
                # image = []
                # for s, duration in grouped_by_shots_sorted_by_depth:
                #     image.append(duration['4'].to_numpy())

                if f"qubits_{q}_depth_{d}" not in figures:
                    figures[f"qubits_{q}_depth_{d}"] = go.Figure()

                figures[f"qubits_{q}_depth_{d}"].add_trace(
                    go.Scatter(
                        name=f"{framework_name}",
                        x=duration_sorted_by_shots["shots"].astype(int),
                        y=durations_mean,
                        mode="lines",
                        line=dict(color=main_color_sel),
                    )
                )
                figures[f"qubits_{q}_depth_{d}"].add_trace(
                    go.Scatter(
                        name=f"{framework_name} - High",
                        x=duration_sorted_by_shots["shots"].astype(int),
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
                        x=duration_sorted_by_shots["shots"].astype(int),
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
                        type=design.shots_tick_type,
                        tickmode="array",
                        tickvals=duration_sorted_by_shots["shots"].astype(int),
                        # ticktext=[
                        #     f"2^{i}"
                        #     for i in duration_sorted_by_shots["qubits"].astype(int)
                        # ],
                        tickangle=design.shots_tickangle,
                        title="Num. of Shots",
                        showgrid=design.showgrid,
                    ),
                    yaxis=dict(
                        title=f"Time ({si_time})",
                        type=design.time_tick_type,
                        dtick=design.time_dtick,
                        showgrid=design.showgrid,
                    ),
                    title=dict(
                        text=f"Duration per Framework over num. of Shots @ {q} Qubits, Circuit Depth {d}"
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
        evaluations_combined.filter(regex=duration_perf_regex)
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

                durations = duration_sorted_by_depth.filter(regex=duration_perf_regex)
                durations *= factor_time

                durations_mean = durations.mean(axis=1)
                durations_max = durations.max(axis=1)
                durations_min = durations.min(axis=1)

                q = int(q)
                s = int(s)
                # image = []
                # for s, duration in grouped_by_shots_sorted_by_depth:
                #     image.append(duration['4'].to_numpy())

                if f"shots_{s}_qubits_{q}" not in figures:
                    figures[f"shots_{s}_qubits_{q}"] = go.Figure()

                figures[f"shots_{s}_qubits_{q}"].add_trace(
                    go.Scatter(
                        name=f"{framework_name}",
                        x=duration_sorted_by_depth["depth"].astype(int),
                        y=durations_mean,
                        mode="lines",
                        line=dict(color=main_color_sel),
                    )
                )
                figures[f"shots_{s}_qubits_{q}"].add_trace(
                    go.Scatter(
                        name=f"{framework_name} - High",
                        x=duration_sorted_by_depth["depth"].astype(int),
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
                        x=duration_sorted_by_depth["depth"].astype(int),
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
                        type=design.depth_tick_type,
                        tickmode="array",
                        tickvals=duration_sorted_by_depth["depth"].astype(int),
                        # ticktext=[
                        #     f"2^{i}"
                        #     for i in duration_sorted_by_depth["qubits"].astype(int)
                        # ],
                        tickangle=design.depth_tickangle,
                        title="Circuit Depth",
                        showgrid=design.showgrid,
                    ),
                    yaxis=dict(
                        title=f"Time ({si_time})",
                        showgrid=design.showgrid,
                        dtick=design.time_dtick,
                        type=design.time_tick_type,
                    ),
                    title=dict(
                        text=f"Duration per Framework over Circuit Depth @ {s} Shots, {q} Qubits"
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


def extract_tests(evaluations_combined: Dict):
    si_time, factor_time = get_time_scale(
        evaluations_combined.filter(regex=duration_perf_regex)
    )

    durations_perf = evaluations_combined.filter(regex=duration_perf_regex)

    durations_proc = evaluations_combined.filter(regex=duration_proc_regex)

    n_runs = durations_perf.shape[0]
    n_evals = durations_perf.shape[1]

    evals_perf_mean = durations_perf.mean(axis=0)
    evals_proc_mean = durations_proc.mean(axis=0)

    log = logging.getLogger(__name__)
    log.info(f"Received {n_evals} evaluations and {n_runs} runs")
    log.info(f"Evaluation mean across all runs (perf timing): {evals_perf_mean.mean()}")
    log.info(
        f"Evaluation deviation across all runs (perf timing): {evals_perf_mean.max() - evals_perf_mean.min()}"
    )
    log.info(f"Evaluation mean across all runs (proc timing): {evals_proc_mean.mean()}")
    log.info(
        f"Evaluation deviation across all runs (proc timing): {evals_proc_mean.max() - evals_proc_mean.min()}"
    )

    return {}


def export_selected(evaluations_combined, additional_figures, output_folder, **figures):
    max_qubits = evaluations_combined["qubits"].max()
    max_depth = evaluations_combined["depth"].max()
    max_shots = evaluations_combined["shots"].max()
    frameworks = evaluations_combined["framework"].unique()

    def export(fig, name, folder):
        pio.full_figure_for_development(
            fig, warn=False
        )  # Disable warnings to prevent printing a box at the bottom left of the figure. See this issue: https://github.com/plotly/plotly.py/issues/3469
        pio.kaleido.scope.mathjax = None  # Disable mathjax completely since above did not help. See this issue: https://github.com/plotly/Kaleido/issues/122

        fig.write_image(os.path.join(folder, f"{name}.pdf"), engine="kaleido")

    sel = f"shots_{max_shots}_depth_{max_depth}"
    export(figures[sel], sel, output_folder)

    sel = f"shots_{max_shots}_qubits_{max_qubits}"
    export(figures[sel], sel, output_folder)

    sel = f"qubits_{max_qubits}_depth_{max_depth}"
    export(figures[sel], sel, output_folder)

    for fw in frameworks:
        sel = f"{fw}_qubits_{max_qubits}"
        export(figures[sel], sel, output_folder)

        sel = f"{fw}_depth_{max_depth}"
        export(figures[sel], sel, output_folder)

        sel = f"{fw}_shots_{max_shots}"
        export(figures[sel], sel, output_folder)

    for add_fig in additional_figures:
        export(figures[add_fig], add_fig, output_folder)

    return {}
