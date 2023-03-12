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
    all_partitions = [Path(f).stem for f in glob.glob("data/02_intermediate/*.csv")]
    existing_durations = [
        Path(f).stem for f in glob.glob("data/05_execution_durations/*.csv")
    ]
    existing_results = [
        Path(f).stem for f in glob.glob("data/04_execution_results/*.csv")
    ]
    existing_measurements = [m for m in existing_durations if m in existing_results]
    partitions = [p for p in all_partitions if p not in existing_measurements]

    figures = [Path(f).stem for f in glob.glob("data/07_reporting/*.tmp")]

    dg_pipelines = dg.create_pipeline(partitions=all_partitions)
    ds_pipelines = ds.create_pipeline(partitions=partitions)
    viz_pipelines = viz.create_pipeline(figures=figures)

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
        # ct measure is the same as the measure pipeline, but we need to tell the hooks that we don't want to delete the existing results
        "ctmeasure": dg_pipelines[
            "pl_generate_qasm_circuits"
            # "pl_generate_qasm_circuits_splitted"
        ]
        + ds_pipelines["pl_parallel_measure_execution_durations"],
        "visualize": viz_pipelines["pl_visualize_evaluations"],
    }
