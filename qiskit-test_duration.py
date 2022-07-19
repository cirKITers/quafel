import qiskit as q
from qiskit.circuit.random import random_circuit
import numpy as np
import matplotlib.pyplot as plt
import time

shots_list = [None, *[2**s for s in range(7, 11)]]
seed = 10000
evals = 20

# -------------------------------------------------------------------------------------

qubits = 2
depth = 10

duration_matrix = np.zeros((len(shots_list), depth))

for i, shots in enumerate(shots_list):
    for d in range(depth):
        if shots == None:
            backend = q.Aer.get_backend("statevector_simulator")
        else:
            backend = q.Aer.get_backend("qasm_simulator")

        qcs = []
        for e in range(evals):
            qc = random_circuit(2, d, max_operands=3, measure=True, seed=seed)
            qc.measure_all()

            qcs.append(qc)

        start_time = time.time()
        job_result = q.execute(qcs, backend=backend, shots=shots).result()
        duration_matrix[i,d] = time.time() - start_time
        # print(f"Execution of {evals} circuits took {duration} seconds.")
    print(f"Progress: {i*depth}/{len(shots_list)*depth}")


fig = plt.figure()
plt.xlabel(f'shots')
plt.ylabel(f'depth')
plt.imshow(duration_matrix)
plt.savefig(f"plots/duration_depth_{qubits}_{depth}.png")

# -------------------------------------------------------------------------------------

qubits = 10
depth = 5

for i, shots in enumerate(shots_list):
    for d in range(qubits):
        if shots == None:
            backend = q.Aer.get_backend("statevector_simulator")
        else:
            backend = q.Aer.get_backend("qasm_simulator")

        qcs = []
        for e in range(evals):
            qc = random_circuit(2, d, max_operands=3, measure=True, seed=seed)
            qc.measure_all()

            qcs.append(qc)

        start_time = time.time()
        job_result = q.execute(qcs, backend=backend, shots=shots).result()
        duration_matrix[i,d] = time.time() - start_time
        # print(f"Execution of {evals} circuits took {duration} seconds.")
    print(f"Progress: {i*depth}/{len(shots_list)*depth}")

# def exp_val(results, shots=None):
#     p_0 = results["0"] if "0" in results else 0
#     p_1 = results["1"] if "1" in results else 0
        

#     shots = 1 if shots == None else shots

#     return 1/shots * (p_1-p_0)

# results = [exp_val(job_result.get_counts(qc), shots) for qc in qcs]
# results = [exp_val(job_result.get_statevector(qc), shots) for qc in qcs]

fig = plt.figure()
plt.xlabel(f'shots')
plt.ylabel(f'depth')
plt.imshow(duration_matrix)
plt.savefig(f"plots/duration_qubits_{qubits}_{depth}.png")