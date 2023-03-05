from kedro.extras.datasets.pandas import CSVDataSet
from kedro.extras.datasets.plotly import JSONDataSet
from kedro.io import Version
from kedro.framework.hooks import hook_impl

from typing import Any, Dict

import glob
from pathlib import Path

import os
import logging
from kedro.io import DataCatalog


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
        # if (
        #     run_params["pipeline_name"] == "parallel"
        #     or run_params["pipeline_name"] == "viz"
        #     or run_params["pipeline_name"] is None
        # ):
        #     # add input dataset
        #     partitions = glob.glob("data/02_intermediate/*.csv")
        #     fw_name = (
        #         catalog.datasets.params__data_science__framework_identifier
        #     )

        #     for i, partition in enumerate(partitions):
        #         # partition loader
        #         evaluation_partitions_name = (
        #             f"data_generation.evaluation_partition_{i}"
        #         )
        #         input_dataset = CSVDataSet(filepath=partition)
        #         catalog.add(evaluation_partitions_name, input_dataset)

        #     evaluation_matrix = (
        #         catalog.datasets.data_generation__evaluation_matrix.load()
        #     )

        #     names = []
        #     for f in evaluation_matrix["frameworks"]:
        #         for q in evaluation_matrix["qubits"]:
        #             names.append(f"{f}_qubits_{q}")
        #         for d in evaluation_matrix["depths"]:
        #             names.append(f"{f}_depth_{d}")

        #     version = Version(
        #         None, catalog.datasets.dummy_versioned_dataset._version.save
        #     )
        #     for name in names:
        #         filepath = os.path.join("data/07_reporting/", f"{name}.json")

        #         dataset_template = JSONDataSet(
        #             filepath=filepath, version=version
        #         )
        #         catalog.add(name, dataset_template)

        #         try:
        #             os.mkdir(filepath)
        #         except FileExistsError:
        #             # directory already exists
        #             pass
        #         # with open(filepath, 'w') as f:
        #         #     f.write('')

        if run_params["pipeline_name"] == "pre":
            tempFiles = glob.glob("data/02_intermediate/*.csv")
            for f in tempFiles:
                os.remove(f)

            tempFiles = glob.glob("data/05_execution_durations/*.csv")
            for f in tempFiles:
                os.remove(f)

            tempFiles = glob.glob("data/07_reporting/*.tmp")
            for f in tempFiles:
                os.remove(f)

            pass  # TODO: delete all the old intermediate results

        # # add output dataset


class DataCatalogHooks:
    @property
    def _logger(self):
        return logging.getLogger(self.__class__.__name__)

    @hook_impl
    def after_catalog_created(self, catalog: DataCatalog) -> None:
        # add input dataset
        partitions = glob.glob("data/02_intermediate/*.csv")

        for partition in partitions:
            # partition loader
            evaluation_partitions_name = (
                f"data_generation.evaluation_partition_{Path(partition).stem}"
            )
            input_dataset = CSVDataSet(filepath=partition)
            catalog.add(evaluation_partitions_name, input_dataset)

        try:
            evaluation_matrix = (
                catalog.datasets.data_generation__evaluation_matrix.load()
            )

            names = []
            for f in evaluation_matrix["frameworks"]:
                for q in evaluation_matrix["qubits"]:
                    names.append(f"{f}_qubits_{q}")
                for d in evaluation_matrix["depths"]:
                    names.append(f"{f}_depth_{d}")

            version = Version(
                None, catalog.datasets.dummy_versioned_dataset._version.save
            )

            for name in names:
                filepath = os.path.join("data/07_reporting/", f"{name}.json")

                dataset_template = JSONDataSet(
                    filepath=filepath, version=version
                )
                catalog.add(name, dataset_template)

                try:
                    os.mkdir(filepath)
                except FileExistsError:
                    # directory already exists
                    pass

                with open(
                    os.path.join("data/07_reporting/", f"{name}.tmp"), "w"
                ) as f:
                    f.write("")
        except:
            pass
