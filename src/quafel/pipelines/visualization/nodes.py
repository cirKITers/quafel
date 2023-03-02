"""
This is a boilerplate pipeline 'visualization'
generated using Kedro 0.18.4
"""

import plotly.express as px
from typing import Dict
import pandas as pd


def shots_depths_viz(evaluation_partitions: Dict, execution_durations: Dict):
    combine_all = pd.DataFrame()

    for (partition_id, partition_load_func), (
        duration_id,
        duration_load_func,
    ) in zip(evaluation_partitions.items(), execution_durations.items()):
        partition_data = partition_load_func()
        duration_data = duration_load_func()

        combined_partition_duration = pd.concat(
            [partition_data, duration_data],
            ignore_index=True,
            sort=True,
            axis=0,
        )

        combine_all = pd.concat(
            [combine_all, combined_partition_duration],
            ignore_index=True,
            sort=True,
            axis=1,
        )

    return combine_all

    pass


def shots_qubits_viz(evaluation_partitions, execution_durations):
    pass
