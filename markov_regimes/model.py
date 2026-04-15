from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from .config import ProcessConfig


@dataclass
class ScenarioModel:
    """Top-level model that owns all processes and runs simulation.

    Attributes:
        processes: list of ProcessConfig, one per independent Markov chain.
    """

    processes: List[ProcessConfig]

    def __post_init__(self) -> None:
        if not self.processes:
            raise ValueError("Must provide at least one process")

    @property
    def total_dim(self) -> int:
        return sum(p.d for p in self.processes)

    def simulate(
        self,
        n_paths: int,
        n_steps: int,
        dt: float = 1.0,
        seed: int | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate the multi-process regime-switching model.

        Args:
            n_paths: number of Monte Carlo paths (M).
            n_steps: number of timesteps (T).
            dt: time increment per step.
            seed: optional RNG seed for reproducibility.

        Returns:
            values: np.ndarray of shape (M, T, total_dim) — process values.
            regimes: np.ndarray of shape (M, T, N) — regime index per process.
        """
        rng = np.random.default_rng(seed)
        N = len(self.processes)
        D = self.total_dim

        values = np.zeros((n_paths, n_steps, D))
        regimes = np.zeros((n_paths, n_steps, N), dtype=np.int32)

        # Precompute cumulative transition matrices for efficient sampling
        cum_trans = [np.cumsum(p.transition_matrix, axis=1) for p in self.processes]

        # Precompute sqrt(dt) and Cholesky-like factors per regime per process
        sqrt_dt = np.sqrt(dt)

        # Initialize: sample initial regime from stationary distribution,
        # set x0 to the corresponding regime's mu (or user-provided x0)
        col = 0
        for j, proc in enumerate(self.processes):
            pi = proc.stationary_distribution
            init_regimes = rng.choice(proc.n_regimes, size=n_paths, p=pi).astype(np.int32)
            regimes[:, 0, j] = init_regimes

            if proc.x0 is not None:
                values[:, 0, col : col + proc.d] = proc.x0[np.newaxis, :]
            else:
                # Each path starts at the mu of its sampled initial regime
                mu_stack = np.array([proc.regimes[r].mu for r in range(proc.n_regimes)])
                values[:, 0, col : col + proc.d] = mu_stack[init_regimes]
            col += proc.d

        # Pre-draw all uniform random numbers for regime transitions
        # Shape: (n_paths, n_steps - 1, N)
        u_regime = rng.uniform(size=(n_paths, n_steps - 1, N))

        # Pre-draw all standard normals for OU diffusion
        # Shape: (n_paths, n_steps - 1, D)
        z = rng.standard_normal(size=(n_paths, n_steps - 1, D))

        for t in range(1, n_steps):
            col = 0
            for j, proc in enumerate(self.processes):
                d = proc.d
                prev_regime = regimes[:, t - 1, j]  # (M,)

                # --- Regime transition ---
                # Gather the cumulative transition row for each path's previous regime
                # cum_trans[j] has shape (R, R); index by prev_regime -> (M, R)
                cum_row = cum_trans[j][prev_regime]  # (M, R)
                u = u_regime[:, t - 1, j : j + 1]  # (M, 1)
                new_regime = (u < cum_row).argmax(axis=1).astype(np.int32)  # (M,)
                regimes[:, t, j] = new_regime

                # --- OU dynamics ---
                x_prev = values[:, t - 1, col : col + d]  # (M, d)
                dw = z[:, t - 1, col : col + d] * sqrt_dt  # (M, d)

                # Vectorised over paths: group by regime to apply correct parameters
                x_next = np.empty_like(x_prev)
                for r in range(proc.n_regimes):
                    mask = new_regime == r
                    if not np.any(mask):
                        continue
                    rc = proc.regimes[r]
                    x_r = x_prev[mask]  # (n_r, d)
                    # drift: K (mu - x) dt
                    drift = (rc.mu[np.newaxis, :] - x_r) @ rc.K.T * dt  # (n_r, d)
                    # diffusion: sigma @ dW
                    diffusion = dw[mask] @ rc.sigma.T  # (n_r, d)
                    x_next[mask] = x_r + drift + diffusion

                values[:, t, col : col + d] = x_next
                col += d

        return values, regimes
