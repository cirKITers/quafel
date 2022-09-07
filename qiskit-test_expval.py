import qiskit as q
import numpy as np
import matplotlib.pyplot as plt
import time


evals = 10
shots = None
seed = 10000

shots_list = [None, *[2**s for s in range(2, 12)]]

for shots in shots_list:
    if shots == None:
        backend = q.Aer.get_backend("statevector_simulator")
    else:
        backend = q.Aer.get_backend("qasm_simulator")

    qcs = []
    for e in range(evals):
        qc = q.QuantumCircuit(1)
        qc.rx(e * np.pi / evals, 0)
        qc.measure_all()

        qcs.append(qc)

    start_time = time.time()
    job_result = q.execute(qcs, backend=backend, shots=shots).result()
    duration = time.time() - start_time
    print(f"Execution of {evals} circuits took {duration} seconds.")

    def exp_val(results, shots=None):
        p_0 = results["0"] if "0" in results else 0
        p_1 = results["1"] if "1" in results else 0

        shots = 1 if shots == None else shots

        return 1 / shots * (p_1 - p_0)

    results = [exp_val(job_result.get_counts(qc), shots) for qc in qcs]

    fig = plt.figure()
    plt.xlabel(f"shots")
    plt.ylabel(f"depth")
    plt.plot(results)
    plt.savefig(f"plots/expval_{evals}_{shots}.png")
