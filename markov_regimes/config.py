from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class RegimeConfig:
    """Parameters for a single regime: mean-reversion matrix K, long-run mean mu, volatility matrix sigma."""

    K: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, dtype=np.float64)
        self.mu = np.asarray(self.mu, dtype=np.float64)
        self.sigma = np.asarray(self.sigma, dtype=np.float64)

    def validate(self, d: int) -> None:
        if self.mu.shape != (d,):
            raise ValueError(f"mu must have shape ({d},), got {self.mu.shape}")
        if self.K.shape != (d, d):
            raise ValueError(f"K must have shape ({d}, {d}), got {self.K.shape}")
        if self.sigma.shape != (d, d):
            raise ValueError(f"sigma must have shape ({d}, {d}), got {self.sigma.shape}")


@dataclass
class ProcessConfig:
    """Configuration for a single process with its own independent Markov chain."""

    name: str
    d: int  # dimension of the process
    transition_matrix: np.ndarray
    regimes: List[RegimeConfig]
    x0: np.ndarray | None = None  # initial state; defaults to mu of regime 0

    def __post_init__(self) -> None:
        self.transition_matrix = np.asarray(self.transition_matrix, dtype=np.float64)
        if self.x0 is not None:
            self.x0 = np.asarray(self.x0, dtype=np.float64)
        self._validate()

    def _validate(self) -> None:
        n_regimes = len(self.regimes)
        if n_regimes == 0:
            raise ValueError(f"Process '{self.name}': must have at least one regime")

        # Transition matrix shape
        if self.transition_matrix.shape != (n_regimes, n_regimes):
            raise ValueError(
                f"Process '{self.name}': transition_matrix must have shape "
                f"({n_regimes}, {n_regimes}), got {self.transition_matrix.shape}"
            )

        # Rows must sum to 1
        row_sums = self.transition_matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            raise ValueError(
                f"Process '{self.name}': transition_matrix rows must sum to 1, "
                f"got row sums {row_sums}"
            )

        # Non-negative entries
        if np.any(self.transition_matrix < 0):
            raise ValueError(
                f"Process '{self.name}': transition_matrix must have non-negative entries"
            )

        # Validate each regime's dimensions
        for i, regime in enumerate(self.regimes):
            try:
                regime.validate(self.d)
            except ValueError as e:
                raise ValueError(f"Process '{self.name}', regime {i}: {e}") from e

        # Validate x0 if provided
        if self.x0 is not None and self.x0.shape != (self.d,):
            raise ValueError(
                f"Process '{self.name}': x0 must have shape ({self.d},), got {self.x0.shape}"
            )

    @property
    def n_regimes(self) -> int:
        return len(self.regimes)

    @property
    def stationary_distribution(self) -> np.ndarray:
        """Compute the stationary distribution pi such that pi @ P = pi.

        Solves the constrained system via eigendecomposition: pi is the left
        eigenvector of P corresponding to eigenvalue 1, normalised to sum to 1.
        """
        P = self.transition_matrix
        # Left eigenvectors: rows of V^T where V comes from eig(P^T)
        eigenvalues, eigenvectors = np.linalg.eig(P.T)
        # Find the eigenvector closest to eigenvalue 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi = np.real(eigenvectors[:, idx])
        pi = pi / pi.sum()
        return pi
