import qiskit as q
from qiskit.circuit.random import random_circuit
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import time
import plot_util
import classes
import config

user_input = input("Choose your framework (pennylane, qiskit or cirq): \n")

# Inhalt der config datei -----------------------------
shots_list = config.shots_list
seed = config.seed
evals = config.evals
qubits = config.qubits
depth = config.depth
framework = getattr(classes, "duration_" + user_input)()
# framework = eval("classes.duration_" + user_input +"()")
# -----------------------------------------------------

duration_matrix_depth = np.zeros((len(shots_list), depth))
duration_matrix_qubits = np.zeros((len(shots_list), depth))

for i, shots in enumerate(shots_list):

    # iteration over depth
    for j in range(1, depth + 1):
        framework.depth = j
        framework.generate_circuit(shots)
        start_time = time.time()
        framework.execute(shots)
        duration_matrix_depth[i, j - 1] = time.time() - start_time

    # iteration over qubits
    for j in range(1, qubits + 1):
        framework.qubits = j
        framework.generate_circuit(shots)
        start_time = time.time()
        framework.execute(shots)
        duration_matrix_qubits[i, j - 1] = time.time() - start_time

    print(f"Progress: {i*depth}/{len(shots_list)*depth}")

# Darstellung depth -----------------------------------------
fig, ax = plt.subplots()
im, cbar = plot_util.heatmap(
    duration_matrix_depth,
    shots_list,
    [d for d in range(depth)],
    ax=ax,
    cmap="magma_r",
    cbarlabel=f"{evals} circuit duration (s) - {qubits} qubits",
    axis_labels=("Circuit Depth", "Analytical (An.) / Number of Shots"),
    title=user_input + " Duration Test - Circuit Depth",
)
texts = plot_util.annotate_heatmap(im, valfmt="{x:.1f} s")
fig.tight_layout()
plt.savefig(f"plots/{user_input}_duration_depth_{qubits}_{depth}.png")


# Darstellung qubits ---------------------------------------
fig, ax = plt.subplots()
im, cbar = plot_util.heatmap(
    duration_matrix_qubits,
    shots_list,
    [d for d in range(depth)],
    ax=ax,
    cmap="magma_r",
    cbarlabel=f"{evals} circuit duration (s) - {qubits} qubits",
    axis_labels=("Circuit Qubits", "Analytical (An.) / Number of Shots"),
    title=user_input + " Duration Test - Circuit Qubits",
)
texts = plot_util.annotate_heatmap(im, valfmt="{x:.1f} s")
fig.tight_layout()
plt.savefig(f"plots/{user_input}_duration_qubits_{qubits}_{depth}.png")
