from qiskit.circuit import ClassicalRegister, QuantumCircuit, CircuitInstruction
from qiskit.circuit import Reset
from qiskit.circuit.library import standard_gates
from qiskit.circuit.exceptions import CircuitError
from qiskit.quantum_info import partial_trace
from qiskit import Aer, execute
import numpy as np

from typing import List, Dict
import pandas as pd

import logging

log = logging.getLogger(__name__)


def log_circuit(qasm_circuit):
    return {"circuit_image": None}


def full_generate_random_qasm_circuits(evaluation_partitions, seed=100):
    qasm_circuits = {}
    n_shots = {}
    frameworks = {}

    for partition_id, partition_load_func in evaluation_partitions.items():
        partition_data = partition_load_func()
        partition_data.index = ["framework", "qubits", "depth", "shots"]

        framework = partition_data[partition_id]["framework"]
        qubits = int(partition_data[partition_id]["qubits"])
        depth = int(partition_data[partition_id]["depth"])
        shots = int(partition_data[partition_id]["shots"])

        qasm_circuits[f"qasm_circuit_{partition_id}"] = generate_random_qasm_circuit(
            qubits, depth, seed
        )["qasm_circuit"]
        n_shots[f"n_shots_{partition_id}"] = shots
        frameworks[f"framework_{partition_id}"] = framework

    return {
        **qasm_circuits,
        **n_shots,
        **frameworks,
    }


def part_generate_random_qasm_circuit(partition, seed=100):
    # TODO: improve this by accessing data by name
    framework = partition[partition.columns[0]][0]
    qubits = int(partition[partition.columns[0]][1])
    depth = int(partition[partition.columns[0]][2])
    shots = int(partition[partition.columns[0]][3])

    result = generate_random_qasm_circuit(qubits, depth, seed)

    return {
        "qasm_circuit": result["qasm_circuit"],
        "n_shots": shots,
        "framework": framework,
        "parameters": result["parameters"],
    }


def _random_circuit(
    num_qubits,
    depth,
    max_operands=2,
    measure=True,
    conditional=False,
    reset=False,
    seed=None,
):
    if num_qubits == 0:
        return QuantumCircuit()
    if max_operands < 1 or max_operands > 4:
        raise CircuitError("max_operands must be between 1 and 4")
    max_operands = max_operands if num_qubits > max_operands else num_qubits

    gates_1q = [
        # (Gate class, number of qubits, number of parameters)
        (standard_gates.IGate, 1, 0),
        (standard_gates.XGate, 1, 0),
        (standard_gates.RZGate, 1, 1),
        (standard_gates.HGate, 1, 0),
        (standard_gates.RXGate, 1, 1),
        (standard_gates.RYGate, 1, 1),
        (standard_gates.SGate, 1, 0),
        (standard_gates.TGate, 1, 0),
        (standard_gates.U2Gate, 1, 2),
        (standard_gates.U3Gate, 1, 3),
        (standard_gates.YGate, 1, 0),
        (standard_gates.ZGate, 1, 0),
    ]
    if reset:
        gates_1q.append((Reset, 1, 0))
    gates_2q = [
        (standard_gates.CXGate, 2, 0),
        (standard_gates.CZGate, 2, 0),
        (standard_gates.SwapGate, 2, 0),
    ]
    gates_3q = [
        (standard_gates.CCXGate, 3, 0),
    ]

    gates = gates_1q.copy()
    if max_operands >= 2:
        gates.extend(gates_2q)
    if max_operands >= 3:
        gates.extend(gates_3q)
    gates = np.array(
        gates,
        dtype=[("class", object), ("num_qubits", np.int16), ("num_params", np.int32)],
    )
    gates_1q = np.array(gates_1q, dtype=gates.dtype)

    qc = QuantumCircuit(num_qubits)

    if measure or conditional:
        cr = ClassicalRegister(num_qubits, "c")
        qc.add_register(cr)

    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.default_rng(seed)

    qubits = np.array(qc.qubits, dtype=object, copy=True)

    # Apply arbitrary random operations in layers across all qubits.
    for layer_number in range(depth):
        # We generate all the randomness for the layer in one go, to avoid many separate calls to
        # the randomisation routines, which can be fairly slow.

        # This reliably draws too much randomness, but it's less expensive than looping over more
        # calls to the rng. After, trim it down by finding the point when we've used all the qubits.
        gate_specs = rng.choice(gates, size=len(qubits))
        cumulative_qubits = np.cumsum(gate_specs["num_qubits"], dtype=np.int16)
        # Efficiently find the point in the list where the total gates would use as many as
        # possible of, but not more than, the number of qubits in the layer.  If there's slack, fill
        # it with 1q gates.
        max_index = np.searchsorted(cumulative_qubits, num_qubits, side="right")
        gate_specs = gate_specs[:max_index]
        slack = num_qubits - cumulative_qubits[max_index - 1]
        if slack:
            gate_specs = np.hstack((gate_specs, rng.choice(gates_1q, size=slack)))

        # For efficiency in the Python loop, this uses Numpy vectorisation to pre-calculate the
        # indices into the lists of qubits and parameters for every gate, and then suitably
        # randomises those lists.
        q_indices = np.empty(len(gate_specs) + 1, dtype=np.int16)
        p_indices = np.empty(len(gate_specs) + 1, dtype=np.int16)
        q_indices[0] = p_indices[0] = 0
        np.cumsum(gate_specs["num_qubits"], out=q_indices[1:])
        np.cumsum(gate_specs["num_params"], out=p_indices[1:])
        # parameters = rng.uniform(0, 2 * np.pi, size=p_indices[-1])
        parameters = ParameterVector(f"p_{layer_number}", p_indices[-1])
        rng.shuffle(qubits)

        # We've now generated everything we're going to need.  Now just to add everything.  The
        # conditional check is outside the two loops to make the more common case of no conditionals
        # faster, since in Python we don't have a compiler to do this for us.
        if conditional and layer_number != 0:
            is_conditional = rng.random(size=len(gate_specs)) < 0.1
            condition_values = rng.integers(
                0, 1 << min(num_qubits, 63), size=np.count_nonzero(is_conditional)
            )
            c_ptr = 0
            for gate, q_start, q_end, p_start, p_end, is_cond in zip(
                gate_specs["class"],
                q_indices[:-1],
                q_indices[1:],
                p_indices[:-1],
                p_indices[1:],
                is_conditional,
            ):
                operation = gate(*parameters[p_start:p_end])
                if is_cond:
                    qc.measure(qc.qubits, cr)
                    # The condition values are required to be bigints, not Numpy's fixed-width type.
                    operation.condition = (cr, int(condition_values[c_ptr]))
                    c_ptr += 1
                qc._append(
                    CircuitInstruction(
                        operation=operation, qubits=qubits[q_start:q_end]
                    )
                )
        else:
            for gate, q_start, q_end, p_start, p_end in zip(
                gate_specs["class"],
                q_indices[:-1],
                q_indices[1:],
                p_indices[:-1],
                p_indices[1:],
            ):
                operation = gate(*parameters[p_start:p_end])
                qc._append(
                    CircuitInstruction(
                        operation=operation, qubits=qubits[q_start:q_end]
                    )
                )

    if measure:
        qc.measure(qc.qubits, cr)

    return qc


def generate_random_qasm_circuit(
    qubits: int, depth: int, seed: int
) -> Dict[str, List[float]]:
    """
    Generate a random quantum circuit as a QASM string and a list of parameters.

    Args:
        qubits: Number of qubits.
        depth: Number of gate layers.
        seed: Random seed.

    Returns:
        A dictionary with the key 'qasm_circuit' containing the QASM string and the key 'parameters' containing a list of parameters.
    """
    qc = _random_circuit(qubits, depth, max_operands=2, measure=True, seed=seed)

    rng = np.random.default_rng(seed)

    # generate all the parameters of the circuit in one go
    parameter_values = rng.uniform(0, 2 * np.pi, size=len(qc.parameters))

    # bind the parameters to the circuit
    bound_circuit = qc.assign_parameters(
        {p: v for p, v in zip(qc.parameters, parameter_values)}
    )

    # return the bound circuit and the parameterizable circuit
    return {"qasm_circuit": bound_circuit.qasm(), "circuit": qc}


def generate_evaluation_matrix(
    min_qubits: int,
    max_qubits: int,
    qubits_increment: int,
    qubits_type: int,
    min_depth: int,
    max_depth: int,
    depth_increment: int,
    depth_type: int,
    min_shots: int,
    max_shots: int,
    shots_increment: int,
    shots_type: int,
    frameworks: List[str],
):
    def generate_ticks(min_t, max_t, inc_t, type_t="linear"):
        if type_t == "linear":
            ticks = [i for i in range(min_t, max_t + inc_t, inc_t)]
        elif "exp" in type_t:
            base = int(type_t.split("exp")[1])
            ticks = [base**i for i in range(min_t, max_t + inc_t, inc_t)]
        else:
            raise ValueError("Unknown base specified and type is not linear")

        return ticks

    qubits = generate_ticks(min_qubits, max_qubits, qubits_increment, qubits_type)
    depths = generate_ticks(min_depth, max_depth, depth_increment, depth_type)
    shots = generate_ticks(min_shots, max_shots, shots_increment, shots_type)

    frameworks = frameworks

    return {
        "evaluation_matrix": {
            "frameworks": frameworks,
            "qubits": qubits,
            "depths": depths,
            "shots": shots,
        }
    }


def generate_evaluation_partitions(evaluation_matrix, skip_combinations):
    partitions = {}
    idx = 0
    for f in evaluation_matrix["frameworks"]:
        if "qubits" in skip_combinations:
            q = max(evaluation_matrix["qubits"])
            for d in evaluation_matrix["depths"]:
                for s in evaluation_matrix["shots"]:
                    partitions[f"{idx}"] = {
                        "framework": f,
                        "qubits": q,
                        "depth": d,
                        "shots": s,
                    }
                    idx += 1
        elif "depth" in skip_combinations:
            d = max(evaluation_matrix["depth"])
            for q in evaluation_matrix["qubtis"]:
                for s in evaluation_matrix["shots"]:
                    partitions[f"{idx}"] = {
                        "framework": f,
                        "qubits": q,
                        "depth": d,
                        "shots": s,
                    }
                    idx += 1
        elif "shots" in skip_combinations:
            s = max(evaluation_matrix["shots"])
            for d in evaluation_matrix["depths"]:
                for q in evaluation_matrix["qubits"]:
                    partitions[f"{idx}"] = {
                        "framework": f,
                        "qubits": q,
                        "depth": d,
                        "shots": s,
                    }
                    idx += 1
        else:
            for q in evaluation_matrix["qubits"]:
                for d in evaluation_matrix["depths"]:
                    for s in evaluation_matrix["shots"]:
                        partitions[f"{idx}"] = {
                            "framework": f,
                            "qubits": q,
                            "depth": d,
                            "shots": s,
                        }
                        idx += 1

    eval_partitions = pd.DataFrame(partitions)
    log.info(f"Generated {eval_partitions.shape[1]} partitions")
    return {"evaluation_partitions": eval_partitions}


def calculate_entangling_capability(qasm_circuit, parameters, samples, seed):
    """
    Calculate the entangling capability of a quantum circuit.
    The strategy is taken from https://doi.org/10.48550/arXiv.1905.10876
    Implementation inspiration from https://obliviateandsurrender.github.io/blogs/expr.html

    Args:
        qasm_circuit (str): The QASM circuit representation as a string.
        parameters (ndarray): The parameters used in the circuit.
        samples (int): The number of samples to generate.
        seed (int): The seed for the random number generator.

    Returns:
        dict: A dictionary containing the entangling capability value.
    """

    def meyer_wallach(circuit, samples, params_shape, precision, rng):
        mw_measure = np.zeros(samples, dtype=complex)

        # outer sum of the MW measure; iterate over set of parameters
        for sample in range(samples):
            # FIXME: unify the range for parameters in the circuit
            # generation method and the sampling here
            params = rng.uniform(0, 2 * np.pi, params_shape)
            bound_circuit = circuit.bind_parameters(params)
            # execute the PQC circuit with the current set of parameters
            result = execute(
                bound_circuit, backend=Aer.get_backend("statevector_simulator")
            ).result()

            # extract the statevector from the simulation result
            U = result.get_statevector(bound_circuit, decimals=precision).data.reshape(
                -1, 1
            )

            # generate a list from [0..num_qubits-1]
            # we need that later to trace out the corresponding qubits
            qb = list(range(circuit.num_qubits))
            # initialize the inner sum which corresponds to the entropy
            entropy = 0

            # inner sum of the MW measure
            for j in range(circuit.num_qubits):
                # density of the jth qubit after tracing out the rest
                density = partial_trace(U, qb[:j] + qb[j + 1 :]).data
                # trace of the density matrix
                entropy += np.trace(density**2)

            # inverse of the normalized entropy is the MW for the current sample of parameters
            mw_measure[sample] = 1 - entropy / circuit.num_qubits

        # final normalization according to formula
        return 2 * np.sum(mw_measure).real / sample

    rng = np.random.default_rng(seed=seed)

    qasm_circuit = QuantumCircuit.from_qasm_str(qasm_circuit)

    # TODO: propagate precision to kedro parameters
    entangling_capability = meyer_wallach(
        circuit=qasm_circuit,
        samples=samples,
        params_shape=parameters.shape,
        precision=5,
        rng=rng,
    )

    return {"entangling_capability": entangling_capability}


def calculate_expressibility(qasm_circuit, parameters, samples, seed):
    """
    Calculate the expressibility of a PQC circuit using a randomized estimation scheme.
    The strategy is taken from https://doi.org/10.48550/arXiv.1905.10876
    Implementation inspiration from https://obliviateandsurrender.github.io/blogs/expr.html

    Args:
        qasm_circuit (str): The QASM string representing the quantum circuit
        parameters (np.ndarray): An array of parameters for the PQC circuit
        samples (int): The number of samples to use for estimation
        seed (int): The seed for the random number generator
    Returns:
        dict: A dictionary containing the expressibility value
    """

    def random_haar_unitary(n_qubits: int, rng) -> np.ndarray:
        """
        Generate a random unitary matrix in the Haar measure

        Args:
            n_qubits (int): The number of qubits in the system

        Returns:
            np.ndarray: A 2^n x 2^n unitary matrix representing a random
                unitary in the Haar measure on the unitary group U(2^n)
        """
        N = 2**n_qubits

        # Generate uniformly sampled random complex numbers
        Z = rng.normal(size=(N, N)) + 1.0j * rng.normal(size=(N, N))
        # Do a QR decomposition
        [Q, R] = np.linalg.qr(Z)
        # Allow D following unitary matrix constraints
        D = np.diag(np.diagonal(R) / np.abs(np.diagonal(R)))
        # Composite the Haar Unitary
        return np.dot(Q, D)

    def haar_integral(n_qubits: int, samples: int, rng) -> np.ndarray:
        """
        Compute the Haar integral for a given number of samples

        Args:
            n_qubits (int): The number of qubits in the system
            samples (int): The number of samples to use for estimation

        Returns:
            np.ndarray: A 2^n x 2^n array representing the max expressibility
        """
        N = 2**n_qubits

        Z = np.zeros((N, N), dtype=complex)

        zero_state = np.zeros(N, dtype=complex)
        zero_state[0] = 1

        # A = np.kron(zero_state, np.ones((1, samples)))
        # Z += np.matmul(A, random_haar_unitary(n_qubits).T.reshape(-1, samples))

        for _ in range(samples):
            A = np.matmul(zero_state, random_haar_unitary(n_qubits, rng)).reshape(-1, 1)
            Z += np.kron(A, A.conj().T)
        return Z / samples

    def pqc_integral(circuit, samples, params_shape, precision, rng):
        """
        Compute the entangling capability of a PQC circuit using a randomized
        estimation scheme

        Args:
            circuit (QuantumCircuit): The PQC circuit to be analyzed
            samples (int): The number of samples to use for estimation
            params_shape (tuple): The shape of the array of parameters to be
                used in the circuit simulation

        Returns:
            np.ndarray: A 2^n x 2^n array representing the expressibility of the PQC circuit, where n is the number of qubits in
                the circuit
        """
        N = 2**circuit.num_qubits
        Z = np.zeros((N, N), dtype=complex)

        for _ in range(samples):
            # FIXME: unify the range for parameters in the circuit
            # generation method and the sampling here
            params = rng.uniform(0, 2 * np.pi, params_shape)
            bound_circuit = circuit.bind_parameters(params)
            # execute the PQC circuit with the current set of parameters
            # ansatz = circuit(params, circuit.num_qubits)
            result = execute(
                bound_circuit, backend=Aer.get_backend("statevector_simulator")
            ).result()

            # extract the statevector from the simulation result
            U = result.get_statevector(bound_circuit, decimals=precision).data.reshape(
                -1, 1
            )

            # accumulate the contribution to the expressibility
            Z += np.kron(U, U.conj().T)  # type: ignore
        return Z / samples

    rng = np.random.default_rng(seed=seed)

    qasm_circuit = QuantumCircuit.from_qasm_str(qasm_circuit)
    n_qubits = qasm_circuit.num_qubits

    # FIXME: the actual value is strongly dependend on the seed (~5-10% deviation)
    # TODO: propagate precision to kedro parameters
    expressibility = np.linalg.norm(
        haar_integral(n_qubits=n_qubits, samples=samples, rng=rng)
        - pqc_integral(
            circuit=qasm_circuit,
            samples=samples,
            params_shape=parameters.shape,
            precision=5,
            rng=rng,
        ),
    )

    return {
        "expressibility": expressibility,
    }
