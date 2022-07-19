import qiskit as q
from qiskit.circuit.random import random_circuit
import numpy as np
import matplotlib.pyplot as plt
import time
import plot_util

shots_list = [None, *[2**s for s in range(7, 12)]]
seed = 10000
evals = 20

# -------------------------------------------------------------------------------------

qubits = 2
depth = 10

duration_matrix = np.zeros((len(shots_list), depth))

for i, shots in enumerate(shots_list):
    for j in range(1,depth+1):
        if shots == None:
            backend = q.Aer.get_backend("statevector_simulator")
        else:
            backend = q.Aer.get_backend("qasm_simulator")

        qcs = []
        for e in range(evals):
            qc = random_circuit(qubits, j, max_operands=3, measure=True, seed=seed)
            qc.measure_all()

            qcs.append(qc)

        start_time = time.time()
        job_result = q.execute(qcs, backend=backend, shots=shots).result()
        duration_matrix[i,j-1] = time.time() - start_time
        # print(f"Execution of {evals} circuits took {duration} seconds.")
    print(f"Progress: {i*depth}/{len(shots_list)*depth}")


fig, ax = plt.subplots()
im, cbar = plot_util.heatmap(duration_matrix, shots_list, [d for d in range(depth)], ax=ax,
                   cmap="magma_r", cbarlabel=f"{evals} circuit duration [s] - {qubits} qubits")
texts = plot_util.annotate_heatmap(im, valfmt="{x:.1f} s")
fig.tight_layout()
plt.savefig(f"plots/duration_depth_{qubits}_{depth}.png")
