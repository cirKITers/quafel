from qiskit.circuit import (
    ClassicalRegister,
    QuantumCircuit,
    CircuitInstruction,
    ParameterVector,
    Reset,
)
from qiskit.circuit.library import standard_gates
from qiskit.circuit.exceptions import CircuitError
from qiskit.quantum_info import partial_trace, random_unitary
from qiskit import execute
from qiskit_aer import StatevectorSimulator
import numpy as np

import jax
import jax.numpy as jnp
from typing import List, Dict, Tuple
import pandas as pd
import logging
import os
import hashlib

log = logging.getLogger(__name__)


def log_circuit(qasm_circuit):
    return {"circuit_image": None}


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
            for q in evaluation_matrix["qubits"]:
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

    # Add row index, because the order of the order
    # of the rows might get scrambled otherwise
    eval_partitions = pd.DataFrame(
        partitions, index=["framework", "qubits", "depth", "shots"]
    )
    log.info(f"Generated {eval_partitions.shape[1]} partitions")
    return {"evaluation_partitions": eval_partitions}


def extract_partition_data(partition):
    # TODO: improve this by accessing data by name
    framework = partition[partition.columns[0]][0]
    qubits = int(partition[partition.columns[0]][1])
    depth = int(partition[partition.columns[0]][2])
    shots = int(partition[partition.columns[0]][3])

    return {
        "qubits": qubits,
        "depth": depth,
        "n_shots": shots,
        "framework": framework,
    }


def qasm_circuit_to_qiskit(qasm_circuit):
    qc = QuantumCircuit.from_qasm_str(qasm_circuit)

    return {
        "qiskit_circuit": qc,
    }


def _random_circuit(
    num_qubits,
    depth,
    max_operands=2,
    conditional=False,
    reset=False,
    seed=None,
):
    """
    <<<<<<< HEAD
        Code partly taken from Qiskit:
        qiskit/circuit/random/utils.py

        Generates a random quantum circuit with the number of qubits
        and circuit depth provided.

        The generating method will favor multi-qubit gates.

        Parameterized gates will have UNBOUND parameters, thus it is
        required to bind them before the circuit can be evaluated.

        Args:
            num_qubits (int): Number of qubits.
            depth (int): Depth of the circuit.
            max_operands (int, optional): Maximum number of operands. Defaults to 2.
            conditional (bool, optional): Cond. Circuit?. Defaults to False.
            reset (bool, optional): Whether to reset the circuit. Defaults to False.
            seed (int, optional): Seed for random number generation. Defaults to None.

        Returns:
            QuantumCircuit: The generated quantum circuit.
    """
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

    # add gates to the overall set, depending on the number of operands
    gates = gates_1q.copy()
    if max_operands >= 2:
        gates.extend(gates_2q)
    if max_operands >= 3:
        gates.extend(gates_3q)
    gates = np.array(
        gates,
        dtype=[("class", object), ("num_qubits", np.int16), ("num_params", np.int32)],
    )

    # generate a numpy array, that we will use later, to fill "gaps"
    gates_1q = np.array(gates_1q, dtype=gates.dtype)

    qc = QuantumCircuit(num_qubits)

    cr = ClassicalRegister(num_qubits, "c")
    qc.add_register(cr)

    rng = np.random.default_rng(seed)

    qubits = np.array(qc.qubits, dtype=object, copy=True)

    # Apply arbitrary random operations in layers across all qubits.
    for layer_number in range(depth):
        # We generate all the randomness for the layer in one go,
        # to avoid many separate calls to
        # the randomisation routines, which can be fairly slow.

        # This reliably draws too much randomness,
        # but it's less expensive than looping over more
        # calls to the rng. After, trim it down by finding the point
        # when we've used all the qubits.
        gate_specs = rng.choice(gates, size=len(qubits))
        cumulative_qubits = np.cumsum(gate_specs["num_qubits"], dtype=np.int16)
        # Efficiently find the point in the list where the total gates
        # would use as many as
        # possible of, but not more than, the number of qubits in the layer.
        # If there's slack, fill
        # it with 1q gates.
        # This will, with favor multi qubit gates, but does not ensure >1 qubit gates
        # being used, especially with a low number of layers
        max_index = np.searchsorted(cumulative_qubits, num_qubits, side="right")
        gate_specs = gate_specs[:max_index]
        slack = num_qubits - cumulative_qubits[max_index - 1]
        if slack:
            gate_specs = np.hstack((gate_specs, rng.choice(gates_1q, size=slack)))

        # For efficiency in the Python loop, this uses Numpy vectorisation to
        # pre-calculate the
        # indices into the lists of qubits and parameters for every gate,
        # and then suitably
        # randomises those lists.
        q_indices = np.empty(len(gate_specs) + 1, dtype=np.int16)
        p_indices = np.empty(len(gate_specs) + 1, dtype=np.int16)
        q_indices[0] = p_indices[0] = 0
        np.cumsum(gate_specs["num_qubits"], out=q_indices[1:])
        np.cumsum(gate_specs["num_params"], out=p_indices[1:])
        # parameters = rng.uniform(0, 2 * np.pi, size=p_indices[-1])
        parameters = ParameterVector(f"p_{layer_number}", p_indices[-1])
        rng.shuffle(qubits)

        # We've now generated everything we're going to need.
        # Now just to add everything.  The
        # conditional check is outside the two loops to make
        # the more common case of no conditionals
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
                    # The condition values are required to be bigints,
                    # not Numpy's fixed-width type.
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

    return qc


def generate_random_qasm_circuit_from_partition(partition, seed):
    partition_data = extract_partition_data(partition)
    return {
        **generate_random_qasm_circuit(
            partition_data["qubits"], partition_data["depth"], seed
        ),
        "n_shots": partition_data["depth"],
        "framework": partition_data["framework"],
    }


def generate_random_qasm_circuit(
    qubits: int, depth: int, seed: int
) -> Dict[str, List[float]]:
    """
    Generate a random quantum circuit and its representation as a QASM string.
    Note that the QASM circuit differs from the original circuit in such a way
    that all qubits are being measured.

    Args:
        qubits: Number of qubits.
        depth: Number of gate layers.
        seed: Random seed.

    Returns:
        A dictionary with the key 'qasm_circuit' containing the QASM string and
        the key 'circuit' containing the Qiskit circuit object.
    """
    qc = _random_circuit(qubits, depth, max_operands=2, seed=seed)

    rng = np.random.default_rng(seed)

    # generate all the parameters of the circuit in one go
    parameter_values = rng.uniform(0, 2 * np.pi, size=len(qc.parameters))

    # bind the parameters to the circuit
    bound_circuit = qc.assign_parameters(
        {p: v for p, v in zip(qc.parameters, parameter_values)}
    )
    # measure all of the bound circuit
    # note that we explicitly NOT use measure_all because this would
    # add a barrier in the qasm representation that is not recognized by some
    # frameworks when translating back later
    bound_circuit.measure(bound_circuit.qubits, *bound_circuit.cregs)

    # return the bound circuit and the parameterizable circuit
    return {"qasm_circuit": bound_circuit.qasm(), "qiskit_circuit": qc}


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


def get_pqc_statevector(
    circuit: QuantumCircuit, params: np.ndarray, backend, precision: int, cache=True
) -> float:
    # Note that this is a jax rng, so it does not matter if we call that multiple times
    bound_circuit = circuit.bind_parameters(params)

    # the qasm representation contains the bound parameters, thus it is ok to hash that
    hs = hashlib.md5(bound_circuit.qasm().encode()).hexdigest()

    if cache:
        name = f"pqc_{hs}.npy"

        cache_folder = ".cache"
        if not os.path.exists(cache_folder):
            os.mkdir(cache_folder)

        file_path = os.path.join(cache_folder, name)

        if os.path.isfile(file_path):
            log.debug(f"Loading PQC statevector from {file_path}")
            loaded_array = np.load(file_path)
            return loaded_array

    # execute the PQC circuit with the current set of parameters
    # ansatz = circuit(params, circuit.num_qubits)
    result = execute(bound_circuit, backend=backend).result()

    # extract the statevector from the simulation result
    U = result.get_statevector(bound_circuit, decimals=precision).data

    if cache:
        log.debug(f"Cached PQC statevector into {file_path}")
        np.save(file_path, U)

    return U


def calculate_entangling_capability(
    circuit: QuantumCircuit,
    samples_per_parameter: int,
    samples_per_qubit: int,
    seed: int,
) -> Dict[str, float]:
    """
    Calculate the entangling capability of a quantum circuit.
    The strategy is taken from https://doi.org/10.48550/arXiv.1905.10876
    Implementation inspiration from
    https://obliviateandsurrender.github.io/blogs/expr.html

    Sanity Check; The following circuit should yield an entangling capability of 1.
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0,1)
    qc.rx(*ParameterVector("x", 1), 0)

    Note that if qubits are being measured, this will return almost zero
    because the statevector simulator cannot work on collapsed states.

    If the circuit doesn't contain parameterizable operations, nan is returned.

    Args:
        circuit (QuantumCircuit): The quantum circuit.
        samples_per_parameter (int): The number of samples to generate.
        seed (int): The seed for the random number generator.

    Returns:
        dict: A dictionary containing the entangling capability value.
    """

    def meyer_wallach(circuit, samples, params_shape, precision, rng):
        if circuit.num_qubits == 1:
            return 0

        mw_measure = np.zeros(samples)

        # FIXME: unify the range for parameters in the circuit
        # generation method and the sampling here
        params = rng.uniform(0, 2 * np.pi, (samples, params_shape))

        # generate a list from [0..num_qubits-1]
        # we need that later to trace out the corresponding qubits
        qb = list(range(circuit.num_qubits))

        backend = StatevectorSimulator(precision="single")
        backend.set_options(
            max_parallel_threads=10,
            max_parallel_experiments=0,
            statevector_parallel_threshold=16,
        )

        # outer sum of the MW measure; iterate over set of parameters
        for i in range(samples):
            U = get_pqc_statevector(circuit, params[i], backend, precision)

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
                entropy += np.trace(density @ density).real

            # fixes accumulating decimals that would otherwise lead to a MW > 1
            entropy = entropy / circuit.num_qubits
            # inverse of the normalized entropy is the MW
            # for the current sample of parameters
            mw_measure[i] = 1 - entropy
        # final normalization according to formula
        return max(0, min((2 * np.sum(mw_measure) / samples), 1))

    rng = np.random.default_rng(seed=seed)

    if len(circuit.parameters) == 0:
        entangling_capability = 0
    else:
        # TODO: propagate precision to kedro parameters
        entangling_capability = meyer_wallach(
            circuit=circuit,
            samples=samples_per_parameter * int(np.log(len(circuit.parameters)) + 1)
            + samples_per_qubit * circuit.num_qubits,
            params_shape=len(circuit.parameters),
            precision=5,
            rng=rng,
        )

    return {"entangling_capability": entangling_capability}


def calculate_expressibility(
    circuit: QuantumCircuit,
    samples_per_parameter: int,
    haar_samples_per_qubit: int,
    seed: int,
) -> Dict[str, float]:
    """
    Calculate the expressibility of a PQC circuit using
    a randomized estimation scheme.
    The strategy is taken from https://doi.org/10.48550/arXiv.1905.10876
    Implementation inspiration from
    https://obliviateandsurrender.github.io/blogs/expr.html

    Args:
        circuit (QuantumCircuit): The PQC circuit to be analyzed
        samples_per_parameter (int): The number of samples to use for estimation
        seed (int): The seed for the random number generator

    Returns:
        Dict[str, float]: A dictionary containing the expressibility value
    """

    def random_haar_unitary(n_qubits: int, rng: np.random.RandomState) -> np.ndarray:
        """
        Generate a random unitary matrix in the Haar measure.
        For details on the QR decomposition, see
        https://doi.org/10.48550/arXiv.math-ph/0609050

        Args:
            n_qubits (int): The number of qubits in the system
            rng (np.random.RandomState): The RandomState object to use for
                generating random numbers

        Returns:
            np.ndarray: A 2^n x 2^n unitary matrix representing a random
                unitary in the Haar measure on the unitary group U(2^n)
        """

        N = 2**n_qubits

        # # Generate uniformly sampled random complex numbers
        # Z = jax.random.normal(rng, (N, N)) + 1.0j * jax.random.normal(rng, (N, N))

        # def f(Z):
        #     # Do a QR decomposition
        #     [Q, R] = jnp.linalg.qr(Z)
        #     # Allow D following unitary matrix constraints
        #     D = jnp.diag(jnp.diagonal(R) / jnp.abs(jnp.diagonal(R)))
        #     # Composite the Haar Unitary
        #     return jnp.dot(Q, D)
        # return jax.jit(f)(Z)

        return random_unitary(
            N,
            int(rng._base_array.real[0]),
        ).data

    def haar_integral(
        n_qubits: int, samples: int, rng: np.random.RandomState
    ) -> np.ndarray:
        """
        Compute the Haar integral for a given number of samples

        Args:
            n_qubits (int): The number of qubits in the system
            samples (int): The number of samples to use for estimation
            rng (np.random.RandomState): The RandomState object to use for
                generating random numbers

        Returns:
            np.ndarray: A 2^n x 2^n array representing the normalized max
                expressibility
        """
        N = 2**n_qubits
        Z = jnp.zeros((N, N), dtype=complex)

        zero_state = jnp.zeros(N, dtype=complex)
        zero_state = zero_state.at[0].set(1)

        def f(zero_state, U):
            A = jnp.matmul(zero_state, U).reshape(-1, 1)
            return jnp.kron(A, A.conj().T)

        jf = jax.jit(f)

        for i in range(samples):
            rng, subkey = jax.random.split(rng)
            U = random_haar_unitary(n_qubits, subkey)

            Z += jf(zero_state, U)

        return Z / samples

    def get_haar_integral(
        n_qubits: int, samples: int, rng: np.random.RandomState, cache=True
    ) -> float:
        if cache:
            name = f"haar_{n_qubits}q_{samples}s.npy"

            cache_folder = ".cache"
            if not os.path.exists(cache_folder):
                os.mkdir(cache_folder)

            file_path = os.path.join(cache_folder, name)

            if os.path.isfile(file_path):
                log.debug(f"Loading Haar integral from {file_path}")
                loaded_array = np.load(file_path)
                return loaded_array

        # Note that this is a jax rng, so it does not matter if we
        # call that multiple times
        array = haar_integral(n_qubits, samples, rng)

        if cache:
            log.debug(f"Cached Haar integral into {file_path}")
            np.save(file_path, array)

        return array

    def pqc_integral(
        circuit: QuantumCircuit,
        samples: int,
        params_shape: Tuple,
        precision: int,
        rng: np.random.default_rng,
    ) -> np.ndarray:
        """
        Compute the entangling capability of a PQC circuit using a randomized
        estimation scheme

        Args:
            circuit (QuantumCircuit): The PQC circuit to be analyzed
            samples (int): The number of samples to use for estimation
            params_shape (tuple): The shape of the array of parameters to be
                used in the circuit simulation
            precision (int): The number of decimals to use when extracting the
                statevector from the simulation result

        Returns:
            np.ndarray: A 2^n x 2^n array representing the expressibility
            of the PQC circuit, where n is the number of qubits in the circuit
        """
        N = 2**circuit.num_qubits
        Z = jnp.zeros((N, N), dtype=complex)

        def f(U):
            return jnp.kron(U, U.conj().T)  # type: ignore

        jf = jax.jit(f)

        backend = StatevectorSimulator(precision="single")
        backend.set_options(
            max_parallel_threads=10,
            max_parallel_experiments=0,
            statevector_parallel_threshold=16,
        )

        # FIXME: unify the range for parameters in the circuit
        # generation method and the sampling here
        params = rng.uniform(0, 2 * np.pi, (samples, params_shape))
        for i in range(samples):
            # extract the statevector from the simulation result
            U = get_pqc_statevector(circuit, params[i], backend, precision).reshape(
                -1, 1
            )

            # accumulate the contribution to the expressibility
            Z += jf(U)

        return Z / samples

    rng = np.random.default_rng(seed=seed)
    jrng = jax.random.key(seed)

    def f(haar, pqc):
        # Note that we use the INVERSE here, because a
        # a LOW distance would actually correspond to a HIGH expressibility
        return 1 - jnp.linalg.norm(haar - pqc)

    jf = jax.jit(f)

    # FIXME: the actual value is strongly dependend on the seed (~5-10% deviation)
    # TODO: propagate precision to kedro parameters
    if len(circuit.parameters) == 0:
        expressibility = 0
    else:
        hi = get_haar_integral(
            n_qubits=circuit.num_qubits, samples=haar_samples_per_qubit, rng=jrng
        )
        pi = pqc_integral(
            circuit=circuit,
            samples=samples_per_parameter * int(np.log(len(circuit.parameters)) + 1)
            + haar_samples_per_qubit * circuit.num_qubits,
            params_shape=len(circuit.parameters),
            precision=5,
            rng=rng,
        )
        expressibility = jf(hi, pi)

    return {
        "expressibility": expressibility,
    }


def calculate_measures(
    circuit: QuantumCircuit,
    samples_per_parameter: int,
    haar_samples_per_qubit: int,
    seed: int,
) -> List[float]:
    expressibility = calculate_expressibility(
        circuit=circuit,
        samples_per_parameter=samples_per_parameter,
        haar_samples_per_qubit=haar_samples_per_qubit,
        seed=seed,
    )["expressibility"]

    entangling_capability = calculate_entangling_capability(
        circuit=circuit,
        samples_per_parameter=samples_per_parameter,
        samples_per_qubit=haar_samples_per_qubit,
        seed=seed,
    )["entangling_capability"]

    return {
        "measure": pd.DataFrame(
            {
                "expressibility": [expressibility],
                "entangling_capability": [entangling_capability],
            }
        )
    }
