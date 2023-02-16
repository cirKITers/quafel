"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from quafel.pipelines import data_generation as dg
from quafel.pipelines import data_science as ds


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    dg_pipelines = dg.create_pipeline()
    ds_pipelines = ds.create_pipeline()

    return {
        "default": dg_pipelines["pl_generate_and_log_circuit"]
        + ds_pipelines["pl_measure_execution_durations"],
    }
