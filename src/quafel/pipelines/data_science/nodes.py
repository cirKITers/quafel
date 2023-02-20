from . import frameworks as fw
import time
import pandas as pd


# def execute_circuit(execute_method:callable, n_shots:int, **kwargs):
#     result = execute_method(shots=n_shots, **kwargs)
#     return result


def aggregate_evaluations(*args):
    return {
        "execution_durations": pd.DataFrame(
            {f"{i}": args[i] for i in range(0, len(args) // 2)}
        ),
        "execution_results": {"_": 0},
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
