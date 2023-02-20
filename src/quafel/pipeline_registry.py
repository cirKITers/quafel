"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from quafel.pipelines import data_generation as dg
from quafel.pipelines import data_science as ds

import glob


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    partitions = glob.glob("data/02_intermediate/*.csv")

    dg_pipelines = dg.create_pipeline(n_partitions=len(partitions))
    ds_pipelines = ds.create_pipeline(n_partitions=len(partitions))

    return {
        "default": dg_pipelines["pl_generate_and_log_circuit"]
        + ds_pipelines["pl_measure_execution_durations"],
        "full": dg_pipelines["pl_generate_evaluation_matrix"],
        "parallel": dg_pipelines["pl_parallel_generate_and_log_circuit"]
        + ds_pipelines["pl_parallel_measure_execution_durations"],
    }
