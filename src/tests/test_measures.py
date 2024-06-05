"""
This module contains an example test.

Tests should be placed in ``src/tests``, in modules that mirror your
project's structure, and in files named test_*.py. They are simply functions
named ``test_*`` which test a unit of logic.

To run the tests, run ``kedro test`` from the project root directory.
"""

from pathlib import Path

import pytest

from kedro.framework.project import settings
from kedro.config import ConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.hooks import _create_hook_manager

from qiskit.circuit import QuantumCircuit, Parameter
from quafel.pipelines.data_generation.nodes import calculate_measures


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


# The tests below are here for the demonstration purpose
# and should be replaced with the ones testing the project
# functionality
class TestMeasures:
    samples_per_parameter = 20
    haar_samples_per_qubit = 60
    seed = 100

    def test_idle_circuit(self):
        qc = QuantumCircuit()

        measures = calculate_measures(
            qc,
            samples_per_parameter=self.samples_per_parameter,
            haar_samples_per_qubit=self.haar_samples_per_qubit,
            seed=self.seed,
        )["measure"]
        print(measures)
        assert measures.iloc[0].expressibility == 0
        assert measures.iloc[0].entangling_capability == 0

    def test_bell_state_circuit(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.rx(Parameter("theta"), 0)

        measures = calculate_measures(
            qc,
            samples_per_parameter=self.samples_per_parameter,
            haar_samples_per_qubit=self.haar_samples_per_qubit,
            seed=self.seed,
        )["measure"]
        print(measures)

        # CNOT gate, so full entanglement
        assert measures.iloc[0].entangling_capability == 1.0

        # Due to the coarse sampling, there is quite a range that we allow
        assert measures.iloc[0].expressibility > 0.4
        assert measures.iloc[0].expressibility < 0.6

    def test_circuit19(self):
        qc = QuantumCircuit(2)
        qc.rx(Parameter("theta_0_0"), 0)
        qc.rz(Parameter("theta_0_1"), 0)
        qc.rx(Parameter("theta_1_0"), 1)
        qc.rz(Parameter("theta_1_1"), 1)
        qc.crx(Parameter("theta_0"), 1, 0)
        qc.crx(Parameter("theta_1"), 0, 1)

        measures = calculate_measures(
            qc,
            samples_per_parameter=self.samples_per_parameter,
            haar_samples_per_qubit=self.haar_samples_per_qubit,
            seed=self.seed,
        )["measure"]
        print(measures)

        # CRX, on average we should be > 0
        assert measures.iloc[0].entangling_capability > 0

        # Expected expr. for circuit 19 is quite high
        assert measures.iloc[0].expressibility > 0.8


if __name__ == "__main__":
    tm = TestMeasures()
    tm.test_idle_circuit()
    tm.test_bell_state_circuit()
    tm.test_circuit19()
