"""Equation 3: recombinase expression as a Langevin equation.

``dR/dt = beta_0 - gamma*R + sigma*R*eta(t)``

with ``<eta(t)> = 0`` and ``<eta(t)eta(t')> = delta(t - t')``.  The noise is
multiplicative and the equation is read in the Ito sense, which is what the
Euler-Maruyama update below implements:

``R_{n+1} = R_n + (beta_0 - gamma*R_n)*dt + sigma*R_n*sqrt(dt)*Z_n``,
``Z_n ~ N(0, 1)`` i.i.d.

``R`` is clamped at 0 after each step; a negative protein concentration is
unphysical and the Ito discretisation can undershoot when ``sigma*sqrt(dt)``
is not small.
"""

from __future__ import annotations

import numpy as np

from parameters import BETA_0, DT, GAMMA, R_INITIAL, R_SS, SEED, T_END


def time_grid(dt: float = DT, t_end: float = T_END) -> np.ndarray:
    """Uniform grid ``0, dt, ..., t_end`` in seconds."""
    n_steps = int(round(t_end / dt))
    return np.arange(n_steps + 1, dtype=float) * dt


def simulate(sigma: float,
             n_traj: int = 1,
             dt: float = DT,
             t_end: float = T_END,
             beta_0: float = BETA_0,
             gamma: float = GAMMA,
             R0: float | None = None,
             seed: int = SEED):
    """Euler-Maruyama integration of Equation 3.

    Parameters
    ----------
    sigma
        Noise amplitude (s^-1/2).  ``sigma = 0`` gives the deterministic
        solution of ``dR/dt = beta_0 - gamma*R``.
    n_traj
        Number of independent cells.
    R0
        Initial recombinase concentration in nM.  Defaults to
        ``parameters.R_INITIAL`` = 0: expression starts from zero and
        accumulates, so the deterministic trace is a rising exponential that
        reaches 44.23 nM at ``t_end`` rather than sitting at ``R_ss``.
    seed
        Seed for ``numpy.random.default_rng``.

    Returns
    -------
    t : ndarray, shape (n_steps + 1,)
    R : ndarray, shape (n_traj, n_steps + 1)
    """
    t = time_grid(dt, t_end)
    n_points = t.size
    R0 = R_INITIAL if R0 is None else R0

    rng = np.random.default_rng(seed)
    R = np.empty((n_traj, n_points), dtype=float)
    R[:, 0] = R0

    sqrt_dt = np.sqrt(dt)
    current = np.full(n_traj, float(R0))
    for i in range(1, n_points):
        if sigma == 0.0:
            noise = 0.0
        else:
            noise = sigma * current * sqrt_dt * rng.standard_normal(n_traj)
        current = current + (beta_0 - gamma * current) * dt + noise
        np.maximum(current, 0.0, out=current)  # clamp: R >= 0
        R[:, i] = current

    return t, R


def deterministic_solution(t, beta_0: float = BETA_0, gamma: float = GAMMA,
                           R0: float | None = None):
    """Closed form of ``dR/dt = beta_0 - gamma*R`` (the ``sigma = 0`` limit).

    ``R(t) = R_ss + (R0 - R_ss)*exp(-gamma*t)``, used to check the integrator.
    With the default ``R0 = 0`` this is ``R_ss*(1 - exp(-gamma*t))``, which
    equals 44.2337 nM at ``t_end``.
    """
    R0 = R_INITIAL if R0 is None else R0
    R_inf = beta_0 / gamma
    return R_inf + (R0 - R_inf) * np.exp(-gamma * np.asarray(t, dtype=float))
