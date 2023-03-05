from . import frameworks as fw
import time
import pandas as pd
from typing import Dict

# def execute_circuit(execute_method:callable, n_shots:int, **kwargs):
#     result = execute_method(shots=n_shots, **kwargs)
#     return result


def aggregate_evaluations(*args):
    aggregated = pd.DataFrame({f"{i}": eval for i, eval in enumerate(args)})

    return {
        "execution_durations": aggregated,
        # "execution_results": pd.DataFrame(
        #     {f"{i}": args[i] for i in range(len(args) // 2, len(args))}
        # ),
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

    framework_instance = framework(
        qasm_circuit=qasm_circuit, n_shots=n_shots, **kwargs
    )

    execution_durations = []
    for eval in range(evaluations):
        start = time.time_ns()
        results = framework_instance.execute()
        end = time.time_ns()
        execution_durations.append(end - start)

    return {
        "execution_duration": execution_durations,
        "execution_result": results,
    }


def aggregate_partitions(*args):
    aggregated = pd.concat(args, axis=1)

    return {"aggregated_partitions": aggregated}


def combine_execution_durations(
    evaluation_partitions: Dict, execution_durations: Dict
):
    combine_all = pd.DataFrame()

    for (partition_id, partition_load_func), (
        duration_id,
        duration_load_func,
    ) in zip(evaluation_partitions.items(), execution_durations.items()):
        partition_data = partition_load_func()
        duration_data = duration_load_func()

        combined_partition_duration = pd.concat(
            [partition_data, duration_data], ignore_index=True, axis=0
        )

        combine_all = pd.concat(
            [combine_all, combined_partition_duration],
            ignore_index=True,
            axis=1,
        )

    combine_all = combine_all.transpose()
    # combine_all = combine_all.sort_values(by=[0, 1, 3]) # sort by framework and shots

    return {"execution_durations_combined": combine_all}
