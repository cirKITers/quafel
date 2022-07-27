import classes

shots_list = [None, *[2**s for s in range(7, 12)]]
seed = 10000
evals = 20
qubits = 10
depth = 10
#framework = classes.duration_pennylane()