"""Tests for RegimeConfig and ProcessConfig validation."""

import numpy as np
import pytest

from markov_regimes.config import RegimeConfig, ProcessConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scalar_regime(K=0.5, mu=0.0, sigma=0.1):
    return RegimeConfig(K=np.array([[K]]), mu=np.array([mu]), sigma=np.array([[sigma]]))


def _scalar_process(name="X", n_regimes=2, **kwargs):
    regimes = [_scalar_regime() for _ in range(n_regimes)]
    tm = np.full((n_regimes, n_regimes), 1.0 / n_regimes)
    return ProcessConfig(name=name, d=1, transition_matrix=tm, regimes=regimes, **kwargs)


# ---------------------------------------------------------------------------
# RegimeConfig
# ---------------------------------------------------------------------------

class TestRegimeConfig:
    def test_accepts_lists(self):
        rc = RegimeConfig(K=[[1.0]], mu=[0.0], sigma=[[0.1]])
        assert rc.K.dtype == np.float64
        assert rc.mu.shape == (1,)

    def test_validate_correct(self):
        rc = _scalar_regime()
        rc.validate(1)  # should not raise

    def test_validate_wrong_mu_shape(self):
        rc = RegimeConfig(K=np.eye(2), mu=np.zeros(3), sigma=np.eye(2))
        with pytest.raises(ValueError, match="mu must have shape"):
            rc.validate(2)

    def test_validate_wrong_K_shape(self):
        rc = RegimeConfig(K=np.eye(3), mu=np.zeros(2), sigma=np.eye(2))
        with pytest.raises(ValueError, match="K must have shape"):
            rc.validate(2)

    def test_validate_wrong_sigma_shape(self):
        rc = RegimeConfig(K=np.eye(2), mu=np.zeros(2), sigma=np.eye(3))
        with pytest.raises(ValueError, match="sigma must have shape"):
            rc.validate(2)


# ---------------------------------------------------------------------------
# ProcessConfig
# ---------------------------------------------------------------------------

class TestProcessConfig:
    def test_valid_scalar_process(self):
        p = _scalar_process()
        assert p.n_regimes == 2
        assert p.d == 1

    def test_valid_vector_process(self):
        regimes = [
            RegimeConfig(K=np.eye(2), mu=np.zeros(2), sigma=np.eye(2))
            for _ in range(3)
        ]
        tm = np.array([[0.8, 0.1, 0.1],
                        [0.1, 0.8, 0.1],
                        [0.1, 0.1, 0.8]])
        p = ProcessConfig(name="V", d=2, transition_matrix=tm, regimes=regimes)
        assert p.n_regimes == 3

    def test_no_regimes(self):
        with pytest.raises(ValueError, match="must have at least one regime"):
            ProcessConfig(name="X", d=1, transition_matrix=np.array([[]]), regimes=[])

    def test_transition_matrix_wrong_shape(self):
        with pytest.raises(ValueError, match="transition_matrix must have shape"):
            ProcessConfig(
                name="X", d=1,
                transition_matrix=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                regimes=[_scalar_regime(), _scalar_regime()],
            )

    def test_transition_matrix_rows_not_summing_to_one(self):
        with pytest.raises(ValueError, match="rows must sum to 1"):
            ProcessConfig(
                name="X", d=1,
                transition_matrix=np.array([[0.5, 0.3], [0.4, 0.4]]),
                regimes=[_scalar_regime(), _scalar_regime()],
            )

    def test_transition_matrix_negative_entries(self):
        with pytest.raises(ValueError, match="non-negative"):
            ProcessConfig(
                name="X", d=1,
                transition_matrix=np.array([[1.5, -0.5], [0.5, 0.5]]),
                regimes=[_scalar_regime(), _scalar_regime()],
            )

    def test_regime_dimension_mismatch(self):
        bad_regime = RegimeConfig(K=np.eye(2), mu=np.zeros(2), sigma=np.eye(2))
        with pytest.raises(ValueError, match="regime 0"):
            ProcessConfig(
                name="X", d=1,
                transition_matrix=np.array([[1.0]]),
                regimes=[bad_regime],
            )

    def test_x0_wrong_shape(self):
        with pytest.raises(ValueError, match="x0 must have shape"):
            _scalar_process(x0=np.array([1.0, 2.0]))

    def test_x0_valid(self):
        p = _scalar_process(x0=np.array([5.0]))
        np.testing.assert_array_equal(p.x0, [5.0])


# ---------------------------------------------------------------------------
# Stationary distribution
# ---------------------------------------------------------------------------

class TestStationaryDistribution:
    def test_single_regime(self):
        p = ProcessConfig(
            name="X", d=1,
            transition_matrix=np.array([[1.0]]),
            regimes=[_scalar_regime()],
        )
        np.testing.assert_allclose(p.stationary_distribution, [1.0])

    def test_symmetric_two_state(self):
        """Symmetric transition matrix -> uniform stationary distribution."""
        p = ProcessConfig(
            name="X", d=1,
            transition_matrix=np.array([[0.7, 0.3], [0.3, 0.7]]),
            regimes=[_scalar_regime(), _scalar_regime()],
        )
        np.testing.assert_allclose(p.stationary_distribution, [0.5, 0.5])

    def test_asymmetric_two_state(self):
        """P = [[0.9, 0.1], [0.3, 0.7]] -> pi = [0.75, 0.25]."""
        p = ProcessConfig(
            name="X", d=1,
            transition_matrix=np.array([[0.9, 0.1], [0.3, 0.7]]),
            regimes=[_scalar_regime(), _scalar_regime()],
        )
        np.testing.assert_allclose(p.stationary_distribution, [0.75, 0.25])

    def test_three_state(self):
        tm = np.array([
            [0.90, 0.05, 0.05],
            [0.10, 0.80, 0.10],
            [0.05, 0.05, 0.90],
        ])
        regimes = [_scalar_regime() for _ in range(3)]
        p = ProcessConfig(name="X", d=1, transition_matrix=tm, regimes=regimes)
        pi = p.stationary_distribution
        # pi @ P = pi
        np.testing.assert_allclose(pi @ tm, pi, atol=1e-12)
        # sums to 1
        np.testing.assert_allclose(pi.sum(), 1.0)
        # all non-negative
        assert np.all(pi >= 0)

    def test_sums_to_one(self):
        p = _scalar_process()
        pi = p.stationary_distribution
        np.testing.assert_allclose(pi.sum(), 1.0)
        assert np.all(pi >= -1e-15)
