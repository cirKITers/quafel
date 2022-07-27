import qiskit as q
from qiskit.circuit.random import random_circuit
import pennylane as qml
from pennylane import numpy as np
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
        dev = qml.device("default.qubit", wires=2, shots=shots)

        @qml.qnode(dev)        
        def circuit(w):
            qml.RandomLayers(weights=w, wires=range(2))
            return [qml.expval(qml.PauliZ(i)) for i in range(2)]

        weights = np.random.rand(j, qubits)
        start_time = time.time()
        for _ in range(evals):
            result = circuit(weights)
        duration_matrix[i,j-1] = time.time() - start_time
        # print(f"Execution of {evals} circuits took {duration} seconds.")
    print(f"Progress: {i*depth}/{len(shots_list)*depth}")


fig, ax = plt.subplots()
im, cbar = plot_util.heatmap(duration_matrix, shots_list, [d for d in range(depth)], ax=ax,
                   cmap="magma_r", cbarlabel=f"{evals} circuit duration (s) - {qubits} qubits",
                   axis_labels=("Circuit Depth", "Analytical (An.) / Number of Shots"),
                   title="Pennylane Duration Test - Circuit Depth")
texts = plot_util.annotate_heatmap(im, valfmt="{x:.1f} s")
fig.tight_layout()
plt.savefig(f"plots/pennylane_duration_depth_{qubits}_{depth}.png")
#plt.savefig(f"testing.png")
plt.close() #TODO: still not able to prevent file handle closed warning