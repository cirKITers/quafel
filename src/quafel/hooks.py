from kedro.extras.datasets.pandas import CSVDataSet
from kedro.io import PartitionedDataSet
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

        if run_params["pipeline_name"] is None:
            # add input dataset
            partitions = glob.glob("data/02_intermediate/*.csv")
            fw_name = (
                catalog.datasets.params__data_science__framework_identifier
            )

            for i, partition in enumerate(partitions):
                # partition loader
                evaluation_partitions_name = (
                    f"data_generation.evaluation_partition_{i}"
                )
                input_dataset = CSVDataSet(filepath=partition)
                catalog.add(evaluation_partitions_name, input_dataset)

            # for fw in catalog.datasets.params__data_generation__frameworks.load():
            #     # execution_durations
            #     execution_duration_partitions_name = (
            #         f"execution_duration_partitions_{fw}"
            #     )
            #     execution_duration_partitions_dataset = PartitionedDataSet(
            #         dataset={
            #             "type":"pandas.CSVDataSet",
            #             "save_args":{
            #                 "index":True
            #             }
            #         },
            #         path=f"data/05_execution_durations/{fw}/",
            #         filename_suffix=".csv"
            #     )
            #     catalog.add(execution_duration_partitions_name, execution_duration_partitions_dataset)

            #     # execution_results
            #     execution_result_partitions_name = (
            #         f"execution_result_partitions_{fw}"
            #     )
            #     execution_result_partitions_dataset = PartitionedDataSet(
            #         dataset={
            #             "type":"pandas.CSVDataSet",
            #             "save_args":{
            #                 "index":True
            #             }
            #         },
            #         path=f"data/04_execution_results/{fw}",
            #         filename_suffix=".csv"
            #     )
            #     catalog.add(execution_result_partitions_name, execution_result_partitions_dataset)

        elif run_params["pipeline_name"] == "pre":
            pass  # TODO: delete all the old intermediate results

        # add output dataset
