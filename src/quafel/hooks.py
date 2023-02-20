from kedro.extras.datasets.pandas import CSVDataSet
from kedro.extras.datasets.text import TextDataSet
from kedro.framework.hooks import hook_impl

from typing import Any, Dict

import glob


class ProjectHooks:
    @hook_impl
    def before_pipeline_run(
        self, run_params: Dict[str, Any], pipeline, catalog
    ):
        """A hook implementation to add a catalog entry
        based on the filename passed to the command line, e.g.:
            kedro run --params=input:iris_1.csv
            kedro run --params=input:iris_2.csv
            kedro run --params=input:iris_3.csv
        """
        # filename = run_params["extra_params"]["input"]

        if run_params["pipeline_name"] == "parallel":
            # add input dataset
            partitions = glob.glob("data/02_intermediate/*.csv")

            for i, partition in enumerate(partitions):
                input_dataset_name = (
                    f"data_generation.evaluation_partition_{i}"
                )
                input_dataset = CSVDataSet(filepath=partition)
                catalog.add(input_dataset_name, input_dataset)

                # output_dataset_name = "execution_durations"
                # output_dataset = TextDataSet(filepath=f"data/08_reporting/{filename}")
                # catalog.add(output_dataset_name, output_dataset)

        # add output dataset
