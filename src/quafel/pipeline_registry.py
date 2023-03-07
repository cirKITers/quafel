"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from quafel.pipelines import data_generation as dg
from quafel.pipelines import data_science as ds
from quafel.pipelines import visualization as viz

import glob
import os
from pathlib import Path


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    partitions = glob.glob("data/02_intermediate/*.csv")
    figures = [Path(f).stem for f in glob.glob("data/07_reporting/*.tmp")]
    figures = [Path(f).stem for f in glob.glob("data/07_reporting/*.tmp")]

    dg_pipelines = dg.create_pipeline(n_partitions=len(partitions))
    ds_pipelines = ds.create_pipeline(n_partitions=len(partitions))
    viz_pipelines = viz.create_pipeline(figures)

    return {
        "__default__": dg_pipelines["pl_generate_evaluation_matrix"]
        + dg_pipelines["pl_parallel_generate_and_log_circuit"]
        + ds_pipelines["pl_parallel_measure_execution_durations"]
        + viz_pipelines["pl_visualize_evaluations"],
        "single": dg_pipelines["pl_generate_and_log_circuit"]
        + ds_pipelines["pl_measure_execution_durations"],
        "pre": dg_pipelines["pl_generate_evaluation_matrix"],
        "parallel": dg_pipelines["pl_parallel_generate_and_log_circuit"]
        + ds_pipelines["pl_parallel_measure_execution_durations"],
        # + viz_pipelines["pl_visualize_evaluations"],
        "viz": viz_pipelines["pl_visualize_evaluations"],
    }
