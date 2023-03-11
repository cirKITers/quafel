from . import frameworks as fw
import time
import pandas as pd
from typing import Dict

# def execute_circuit(execute_method:callable, n_shots:int, **kwargs):
#     result = execute_method(shots=n_shots, **kwargs)
#     return result


def aggregate_evaluations(*args):
    aggregated_evaluations = pd.DataFrame({f"{i}": eval for i, eval in enumerate(args)})

    return {
        "aggregated_evaluations": aggregated_evaluations,
    }


def measure_execution_durations(
    evaluations: int,
    qasm_circuit: any,
    n_shots: int,
    framework_id: str,
    **kwargs,
):
    try:
        framework = getattr(fw, framework_id)
    except AttributeError:
        raise AttributeError(
            f"Framework identifier does not match one of the existing frameworks. Existing frameworks are {fw}"
        )

    framework_instance = framework(qasm_circuit=qasm_circuit, n_shots=n_shots, **kwargs)

    execution_durations = []
    execution_results = []
    for eval in range(evaluations):
        start = time.perf_counter()
        framework_instance.execute()
        end = time.perf_counter()
        execution_durations.append(end - start)
        execution_results.append(framework_instance.get_result())

    return {
        "execution_duration": execution_durations,
        "execution_result": execution_results,
    }


def aggregate_partitions(*args):
    aggregated = pd.concat(args, axis=1)

    return {"aggregated_partitions": aggregated}


def combine_evaluations(
    evaluation_partitions: Dict,
    execution_durations: Dict,
    execution_results: Dict,
):
    combine_all = pd.DataFrame()

    for (
        (partition_id, partition_load_func),
        (duration_id, duration_load_func),
        (result_id, result_load_func),
    ) in zip(
        evaluation_partitions.items(),
        execution_durations.items(),
        execution_results.items(),
    ):
        assert (partition_id == duration_id) and (partition_id == result_id)
        partition_data = partition_load_func()
        duration_data = duration_load_func()
        result_data = result_load_func()

        # TODO: unify somehow with the generation part
        partition_data.index = ["framework", "qubits", "depth", "shots"]

        duration_data.index = [f"duration_{i}" for i in range(len(duration_data))]

        result_data.index = [f"result_{i}" for i in range(len(duration_data))]

        combined_partition_duration = pd.concat(
            [partition_data, duration_data, result_data],
            ignore_index=False,
            axis=0,
        )

        combine_all = pd.concat(
            [combine_all, combined_partition_duration],
            ignore_index=False,
            axis=1,
        )

    combine_all = combine_all.transpose()
    # combine_all = combine_all.sort_values(by=[0, 1, 3]) # sort by framework and shots

    return {"evaluations_combined": combine_all}
