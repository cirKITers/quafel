"""
This module contains an example test.

Tests should be placed in ``src/tests``, in modules that mirror your
project's structure, and in files named test_*.py. They are simply functions
named ``test_*`` which test a unit of logic.

To run the tests, run ``kedro test`` from the project root directory.
"""

from pathlib import Path

import pytest
import math

from kedro.framework.project import settings
from kedro.config import ConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.hooks import _create_hook_manager

from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector
from quafel.pipelines.data_generation.nodes import calculate_measures

import logging

log = logging.getLogger(__name__)


@pytest.fixture
def config_loader():
    return ConfigLoader(conf_source=str(Path.cwd() / settings.CONF_SOURCE))


@pytest.fixture
def project_context(config_loader):
    return KedroContext(
        package_name="quafel",
        project_path=Path.cwd(),
        config_loader=config_loader,
        hook_manager=_create_hook_manager(),
    )


class TestMeasures:
    samples_per_parameter = 40
    haar_samples_per_qubit = 60
    seed = 100

    def build_circuit_19(self, n_qubits: int, n_layers: int) -> QuantumCircuit:
        qc = QuantumCircuit(n_qubits)
        log.info(f"Testing Circuit19 with {n_qubits} qubits and {n_layers} layers")

        # Build Circuit19
        for l in range(n_layers):
            # Create Param vector with 3 params per qubit
            w = ParameterVector(f"p_{l}", 3 * n_qubits)
            w_idx = 0
            for q in range(n_qubits):
                qc.rx(w[w_idx], q)
                w_idx += 1
                qc.rz(w[w_idx], q)
                w_idx += 1

            for q in range(n_qubits):
                qc.crx(w[w_idx], n_qubits - q - 1, (n_qubits - q) % n_qubits)
                w_idx += 1

        return qc

    def test_idle_circuit(self):
        qc = QuantumCircuit()

        measures = calculate_measures(
            qc,
            samples_per_parameter=self.samples_per_parameter,
            haar_samples_per_qubit=self.haar_samples_per_qubit,
            seed=self.seed,
        )["measure"]

        assert measures.iloc[0].expressibility == 0
        assert math.isclose(measures.iloc[0].entangling_capability, 0.0, abs_tol=1e-3)

    def test_bell_state_circuit(self):
        # Build Bell State circuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        # Add a single parameter to get "some" expressibility
        qc.rx(Parameter("theta"), 0)

        measures = calculate_measures(
            qc,
            samples_per_parameter=self.samples_per_parameter,
            haar_samples_per_qubit=self.haar_samples_per_qubit,
            seed=self.seed,
        )["measure"]

        # CNOT gate, so full entanglement
        assert math.isclose(measures.iloc[0].entangling_capability, 1.0, abs_tol=1e-3)

        # Due to the coarse sampling, there is quite a range that we allow
        assert measures.iloc[0].expressibility > 0.4
        assert measures.iloc[0].expressibility < 0.6

    def test_circuit19(self):
        n_qubits = 4
        n_layers = 4

        qc = self.build_circuit_19(n_qubits, n_layers)

        measures = calculate_measures(
            qc,
            samples_per_parameter=self.samples_per_parameter,
            haar_samples_per_qubit=self.haar_samples_per_qubit,
            seed=self.seed,
        )["measure"]

        # mean entangling capability according to Eq. (22)
        # in https://doi.org/10.48550/arXiv.1905.10876
        ent_cap_haar_mean = (2**n_qubits - 2) / (2**n_qubits + 1)

        # TODO: this is not very precise yet
        assert math.isclose(
            measures.iloc[0].entangling_capability, ent_cap_haar_mean, abs_tol=0.1
        )

        # Expected expr. for circuit 19 is quite high
        assert measures.iloc[0].expressibility > 0.8

    def test_variance(self):
        n_qubits = 4
        n_layers = 4
        n_samples = 10  # Number of iterations where we calculate the measures

        qc = self.build_circuit_19(n_qubits, n_layers)

        measures = None
        for i in range(n_samples):
            m = calculate_measures(
                qc,
                samples_per_parameter=self.samples_per_parameter,
                haar_samples_per_qubit=self.haar_samples_per_qubit,
                seed=self.seed + i,
            )["measure"]

            measures = measures.append(m) if measures is not None else m

        variance = measures.std()

        assert math.isclose(variance.expressibility, 0.0, abs_tol=0.01)
        assert math.isclose(variance.entangling_capability, 0.0, abs_tol=0.01)


# main for debugging purposes
if __name__ == "__main__":
    tm = TestMeasures()
    tm.test_idle_circuit()
    tm.test_bell_state_circuit()
    tm.test_circuit19()
    tm.test_variance()
