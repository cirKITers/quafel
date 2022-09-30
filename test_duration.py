import numpy as np
import matplotlib.pyplot as plt
import time
import plot_util
import classes
import config
import argparse
import os
import pickle

# user input
CLI = argparse.ArgumentParser()
CLI.add_argument("framework", choices=["pennylane", "qiskit", "cirq", "real"])
CLI.add_argument("--resume", action='store_true')
options = CLI.parse_args()
user_input = options.framework



# Inhalt der config datei -----------------------------
shots_list = config.shots_list
seed = config.seed
evals = config.evals
qubits = config.qubits
depth = config.depth
framework = getattr(classes, "duration_" + user_input).from_config()
# -----------------------------------------------------

if options.resume:
    with open(os.path.join("artifacts", "duration_matrix_depth.pkl"), mode="rb") as f:
        duration_matrix_depth = pickle.load(f)

    with open(os.path.join("artifacts", "duration_matrix_qubits.pkl"), mode="rb") as f:
        duration_matrix_qubits = pickle.load(f)
else:
    duration_matrix_depth = np.ones((len(shots_list), depth))*(-1)
    duration_matrix_qubits = np.ones((len(shots_list), qubits))*(-1)

try:
    for i, shots in enumerate(shots_list):

        # iteration over depth
        for j in range(1, depth + 1):
            if duration_matrix_depth[i, j - 1] != -1:
                continue
            framework.depth = j
            framework.generate_circuit(shots)
            start_time = time.time()
            duration = framework.execute(shots)
            duration_matrix_depth[i, j - 1] = duration if duration is not None else time.time() - start_time


        # iteration over qubits
        for j in range(1, qubits + 1):
            if duration_matrix_qubits[i, j - 1] != -1:
                continue
            framework.qubits = j
            framework.generate_circuit(shots)
            start_time = time.time()
            duration = framework.execute(shots)
            duration_matrix_qubits[i, j - 1] = duration if duration is not None else time.time() - start_time

        print(f"Progress: {i*(depth+qubits)}/{len(shots_list)*(depth+qubits)}")

except KeyboardInterrupt:
    print(f"Interrupted by user, trying to plot what has been measured so far.")
except Exception as e:
    print(f"An error occurred: {e}")

result = input("Save measurement results? [Y/n]")
if result.lower() != "n":
    with open(os.path.join("artifacts", "duration_matrix_depth.pkl"), mode="wb") as f:
        pickle.dump(duration_matrix_depth, f)

    with open(os.path.join("artifacts", "duration_matrix_qubits.pkl"), mode="wb") as f:
        pickle.dump(duration_matrix_qubits, f)


# Darstellung depth -----------------------------------------
fig, ax = plt.subplots()
im, cbar = plot_util.heatmap(
    duration_matrix_depth,
    shots_list,
    [d+1 for d in range(depth)],
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
    [d+1 for d in range(qubits)],
    ax=ax,
    cmap="magma_r",
    cbarlabel=f"{evals} circuit duration (s) - {qubits} qubits",
    axis_labels=("Circuit Qubits", "Analytical (An.) / Number of Shots"),
    title=user_input + " Duration Test - Circuit Qubits",
)
texts = plot_util.annotate_heatmap(im, valfmt="{x:.1f} s")
fig.tight_layout()
plt.savefig(f"plots/{user_input}_duration_qubits_{qubits}_{depth}.png")
