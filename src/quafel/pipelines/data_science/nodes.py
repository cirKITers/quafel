from . import frameworks as fw
import time

# def execute_circuit(execute_method:callable, n_shots:int, **kwargs):
#     result = execute_method(shots=n_shots, **kwargs)
#     return result



def measure_execution_durations(evaluations:int, qasm_circuit:any, framework_identifier:str, n_shots:int, **kwargs):
    try:
        framework = getattr(fw, framework_identifier)
    except AttributeError:
        raise AttributeError(f"Framework identifier does not match one of the existing frameworks. Existing frameworks are {fw}")

    framework_instance = framework(qasm_circuit=qasm_circuit, n_shots=n_shots, **kwargs)

    execution_durations = []
    for eval in range(evaluations):
        start = time.time_ns()
        results = framework_instance.execute()
        end = time.time_ns()
        execution_durations.append(end-start)

    return {
        "execution_durations":execution_durations,
        "execution_results":results
    }