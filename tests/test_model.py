"""Tests for ScenarioModel simulation."""

import numpy as np
import pytest

from markov_regimes.config import RegimeConfig, ProcessConfig
from markov_regimes.model import ScenarioModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scalar_regime(K=0.5, mu=0.0, sigma=0.1):
    return RegimeConfig(K=np.array([[K]]), mu=np.array([mu]), sigma=np.array([[sigma]]))


def _scalar_process(name="X", regimes=None, tm=None):
    if regimes is None:
        regimes = [_scalar_regime(), _scalar_regime(K=1.0, mu=1.0, sigma=0.2)]
    n = len(regimes)
    if tm is None:
        tm = np.full((n, n), 1.0 / n)
    return ProcessConfig(name=name, d=1, transition_matrix=tm, regimes=regimes)


# ---------------------------------------------------------------------------
# ScenarioModel construction
# ---------------------------------------------------------------------------

class TestScenarioModelInit:
    def test_empty_processes(self):
        with pytest.raises(ValueError, match="at least one process"):
            ScenarioModel(processes=[])

    def test_total_dim(self):
        p1 = _scalar_process("A")
        p2 = ProcessConfig(
            name="B", d=2,
            transition_matrix=np.array([[1.0]]),
            regimes=[RegimeConfig(K=np.eye(2), mu=np.zeros(2), sigma=np.eye(2))],
        )
        model = ScenarioModel(processes=[p1, p2])
        assert model.total_dim == 3


# ---------------------------------------------------------------------------
# Output shapes
# ---------------------------------------------------------------------------

class TestSimulationShapes:
    def test_single_scalar_process(self):
        model = ScenarioModel(processes=[_scalar_process()])
        vals, regs = model.simulate(n_paths=50, n_steps=20, seed=0)
        assert vals.shape == (50, 20, 1)
        assert regs.shape == (50, 20, 1)

    def test_multi_process(self):
        p1 = _scalar_process("A")
        p2 = ProcessConfig(
            name="B", d=2,
            transition_matrix=np.array([[0.9, 0.1], [0.2, 0.8]]),
            regimes=[
                RegimeConfig(K=np.eye(2), mu=np.zeros(2), sigma=0.1 * np.eye(2)),
                RegimeConfig(K=np.eye(2), mu=np.ones(2), sigma=0.2 * np.eye(2)),
            ],
        )
        model = ScenarioModel(processes=[p1, p2])
        vals, regs = model.simulate(n_paths=100, n_steps=30, seed=1)
        assert vals.shape == (100, 30, 3)
        assert regs.shape == (100, 30, 2)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_output(self):
        model = ScenarioModel(processes=[_scalar_process()])
        v1, r1 = model.simulate(n_paths=10, n_steps=20, seed=42)
        v2, r2 = model.simulate(n_paths=10, n_steps=20, seed=42)
        np.testing.assert_array_equal(v1, v2)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seed_different_output(self):
        model = ScenarioModel(processes=[_scalar_process()])
        v1, _ = model.simulate(n_paths=10, n_steps=20, seed=1)
        v2, _ = model.simulate(n_paths=10, n_steps=20, seed=2)
        assert not np.array_equal(v1, v2)


# ---------------------------------------------------------------------------
# Deterministic process (sigma=0)
# ---------------------------------------------------------------------------

class TestDeterministicProcess:
    def test_sigma_zero_converges_to_mu(self):
        """With no diffusion and strong mean reversion, process should converge to mu."""
        mu_val = 3.0
        regime = _scalar_regime(K=5.0, mu=mu_val, sigma=0.0)
        proc = ProcessConfig(
            name="det", d=1,
            transition_matrix=np.array([[1.0]]),
            regimes=[regime],
            x0=np.array([0.0]),
        )
        model = ScenarioModel(processes=[proc])
        vals, _ = model.simulate(n_paths=10, n_steps=500, dt=0.01, seed=0)
        # After 500 steps with K=5 and dt=0.01, should be very close to mu
        final = vals[:, -1, 0]
        np.testing.assert_allclose(final, mu_val, atol=0.01)


# ---------------------------------------------------------------------------
# Regime transitions
# ---------------------------------------------------------------------------

class TestRegimeTransitions:
    def test_absorbing_regime(self):
        """If transition matrix is identity, each path stays in its initial regime."""
        regime0 = _scalar_regime(mu=0.0)
        regime1 = _scalar_regime(mu=1.0)
        proc = ProcessConfig(
            name="abs", d=1,
            transition_matrix=np.eye(2),
            regimes=[regime0, regime1],
        )
        model = ScenarioModel(processes=[proc])
        _, regs = model.simulate(n_paths=100, n_steps=50, seed=0)
        # Every timestep should equal the initial regime for that path
        init = regs[:, 0, 0]
        for t in range(1, 50):
            np.testing.assert_array_equal(regs[:, t, 0], init)

    def test_regime_values_in_range(self):
        """All regime indices should be in [0, n_regimes)."""
        proc = _scalar_process()
        model = ScenarioModel(processes=[proc])
        _, regs = model.simulate(n_paths=200, n_steps=100, seed=0)
        assert regs.min() >= 0
        assert regs.max() < proc.n_regimes

    def test_transition_frequencies_approximate(self):
        """With many paths, observed transition frequencies should roughly match the matrix."""
        tm = np.array([[0.9, 0.1], [0.3, 0.7]])
        proc = ProcessConfig(
            name="freq", d=1,
            transition_matrix=tm,
            regimes=[_scalar_regime(), _scalar_regime()],
        )
        model = ScenarioModel(processes=[proc])
        _, regs = model.simulate(n_paths=5000, n_steps=200, seed=42)
        r = regs[:, :, 0]  # (5000, 200)

        # Count transitions from regime 0 -> 0 and 0 -> 1
        prev = r[:, :-1].ravel()
        nxt = r[:, 1:].ravel()

        from_0 = prev == 0
        n_from_0 = from_0.sum()
        if n_from_0 > 0:
            p00 = ((prev == 0) & (nxt == 0)).sum() / n_from_0
            p01 = ((prev == 0) & (nxt == 1)).sum() / n_from_0
            np.testing.assert_allclose(p00, 0.9, atol=0.02)
            np.testing.assert_allclose(p01, 0.1, atol=0.02)

        from_1 = prev == 1
        n_from_1 = from_1.sum()
        if n_from_1 > 0:
            p10 = ((prev == 1) & (nxt == 0)).sum() / n_from_1
            p11 = ((prev == 1) & (nxt == 1)).sum() / n_from_1
            np.testing.assert_allclose(p10, 0.3, atol=0.02)
            np.testing.assert_allclose(p11, 0.7, atol=0.02)


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

class TestInitialConditions:
    def test_default_x0_matches_sampled_regime_mu(self):
        """Without x0, each path's initial value should equal mu of its sampled regime."""
        regimes = [_scalar_regime(mu=10.0), _scalar_regime(mu=-10.0)]
        proc = ProcessConfig(
            name="ic", d=1,
            transition_matrix=np.array([[0.5, 0.5], [0.5, 0.5]]),
            regimes=regimes,
        )
        model = ScenarioModel(processes=[proc])
        vals, regs = model.simulate(n_paths=200, n_steps=2, seed=0)
        init_regime = regs[:, 0, 0]
        expected_mu = np.where(init_regime == 0, 10.0, -10.0)
        np.testing.assert_array_equal(vals[:, 0, 0], expected_mu)

    def test_single_regime_x0_is_mu(self):
        mu_val = 7.0
        regime = _scalar_regime(mu=mu_val)
        proc = ProcessConfig(
            name="ic", d=1,
            transition_matrix=np.array([[1.0]]),
            regimes=[regime],
        )
        model = ScenarioModel(processes=[proc])
        vals, _ = model.simulate(n_paths=5, n_steps=2, seed=0)
        np.testing.assert_array_equal(vals[:, 0, 0], mu_val)

    def test_custom_x0(self):
        proc = ProcessConfig(
            name="ic", d=1,
            transition_matrix=np.array([[1.0]]),
            regimes=[_scalar_regime()],
            x0=np.array([99.0]),
        )
        model = ScenarioModel(processes=[proc])
        vals, _ = model.simulate(n_paths=5, n_steps=2, seed=0)
        np.testing.assert_array_equal(vals[:, 0, 0], 99.0)

    def test_initial_regime_frequencies_match_stationary(self):
        """Initial regime distribution across many paths should match stationary dist."""
        tm = np.array([[0.9, 0.1], [0.3, 0.7]])
        proc = ProcessConfig(
            name="freq", d=1,
            transition_matrix=tm,
            regimes=[_scalar_regime(), _scalar_regime()],
        )
        model = ScenarioModel(processes=[proc])
        _, regs = model.simulate(n_paths=10000, n_steps=2, seed=42)
        init = regs[:, 0, 0]
        freq_0 = (init == 0).mean()
        # Stationary: pi = [0.75, 0.25]
        np.testing.assert_allclose(freq_0, 0.75, atol=0.02)


# ---------------------------------------------------------------------------
# Vector process
# ---------------------------------------------------------------------------

class TestVectorProcess:
    def test_2d_deterministic_convergence(self):
        """2D process with sigma=0 should converge to mu vector."""
        mu = np.array([1.0, -2.0])
        regime = RegimeConfig(
            K=3.0 * np.eye(2),
            mu=mu,
            sigma=np.zeros((2, 2)),
        )
        proc = ProcessConfig(
            name="v2", d=2,
            transition_matrix=np.array([[1.0]]),
            regimes=[regime],
            x0=np.array([0.0, 0.0]),
        )
        model = ScenarioModel(processes=[proc])
        vals, _ = model.simulate(n_paths=5, n_steps=500, dt=0.01, seed=0)
        np.testing.assert_allclose(vals[:, -1, 0], 1.0, atol=0.01)
        np.testing.assert_allclose(vals[:, -1, 1], -2.0, atol=0.01)
