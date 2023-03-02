from kedro.extras.datasets.pandas import CSVDataSet
from kedro.extras.datasets.plotly import JSONDataSet
from kedro.io import Version
from kedro.framework.hooks import hook_impl

from typing import Any, Dict

import glob
import os


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

        if (
            run_params["pipeline_name"] == "parallel"
            or run_params["pipeline_name"] == "viz"
            or run_params["pipeline_name"] is None
        ):
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

            evaluation_matrix = (
                catalog.datasets.data_generation__evaluation_matrix.load()
            )

            names = []
            for f in evaluation_matrix["frameworks"]:
                for q in evaluation_matrix["qubits"]:
                    names.append(f"{f}_qubits_{q}")
                for d in evaluation_matrix["depths"]:
                    names.append(f"{f}_depth_{d}")

            version = Version(None, None)
            for name in names:
                filepath = os.path.join("data/07_reporting/", f"{name}.json")

                dataset_template = JSONDataSet(filepath=filepath)
                catalog.add(name, dataset_template)

                with open(filepath, "w") as f:
                    f.write("")

        elif run_params["pipeline_name"] == "pre":
            pass  # TODO: delete all the old intermediate results

        # add output dataset
