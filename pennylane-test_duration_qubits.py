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

qubits = 10
depth = 5

duration_matrix = np.zeros((len(shots_list), qubits))

for i, shots in enumerate(shots_list):
    for j in range(1,qubits+1):
        dev = qml.device("default.qubit", wires=j, shots=shots)

       
        @qml.qnode(dev)        
        def circuit(w):
            qml.RandomLayers(weights=w, wires=range(j))
            return [qml.expval(qml.PauliZ(i)) for i in range(j)]
        
        weights = np.random.rand(depth, j)
        start_time = time.time()
        for _ in range(evals):    
            result = circuit(weights)
        duration_matrix[i,j-1] = time.time() - start_time
        
        # print(f"Execution of {evals} circuits took {duration} seconds.")
    print(f"Progress: {i*qubits}/{len(shots_list)*qubits}")


fig, ax = plt.subplots()
im, cbar = plot_util.heatmap(duration_matrix, shots_list, [d for d in range(qubits)], ax=ax,
                   cmap="magma_r", cbarlabel=f"{evals} circuit duration [s] - {depth} depth")
texts = plot_util.annotate_heatmap(im, valfmt="{x:.1f} s")
fig.tight_layout()
plt.savefig(f"plots/pennylane_duration_qubits_{qubits}_{depth}.png")