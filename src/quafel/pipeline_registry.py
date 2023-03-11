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

    dg_pipelines = dg.create_pipeline(n_partitions=len(partitions))
    ds_pipelines = ds.create_pipeline(n_partitions=len(partitions))
    viz_pipelines = viz.create_pipeline(figures)

    return {
        "__default__": dg_pipelines["pl_generate_evaluation_partitions"]
        + dg_pipelines["pl_generate_qasm_circuits"]
        + ds_pipelines["pl_parallel_measure_execution_durations"]
        + viz_pipelines["pl_visualize_evaluations"],
        "prepare": dg_pipelines["pl_generate_evaluation_partitions"],
        "measure": dg_pipelines[
            "pl_generate_qasm_circuits"
            # "pl_generate_qasm_circuits_splitted"
        ]
        + ds_pipelines["pl_parallel_measure_execution_durations"],
        "visualize": viz_pipelines["pl_visualize_evaluations"],
    }
