"""Equations 1 and 2 and the five-state master equation.

State vector ordering used everywhere in this repository::

    y = [p_0, p_IA, p_IB, p_A, p_B]

``0``      unbound att site
``IA``/``IB``  recombinase bound at the inversion / excision site (mutually
               exclusive intermediates)
``A``/``B``    inverted / excised DNA; both are absorbing
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from parameters import Parameters, default_parameters

#: Index of each state in the probability vector.
STATE_NAMES = ("p_0", "p_IA", "p_IB", "p_A", "p_B")
I_0, I_IA, I_IB, I_A, I_B = range(5)


# ==========================================================================
# Equation 1 - outcome ratio at steady state
# ==========================================================================

def commitment_probability(k_cat: float, k_off: float) -> float:
    """``a_X = k_cat_X / (k_off_X + k_cat_X)``.

    Probability that a bound recombinase catalyses rather than dissociating.
    """
    return k_cat / (k_off + k_cat)


def phi(p: Parameters | None = None) -> float:
    """Equation 1: the outcome ratio ``P_A / P_B``.

    ``phi = (k_cat_A*(k_off_B + k_cat_B)) / (k_cat_B*(k_off_A + k_cat_A))``

    The recombinase concentration ``R`` does not appear.  Both paths are
    entered from state 0 at the same rate ``k_on*R``, so that rate cancels and
    the ratio is set purely by what happens after binding.
    """
    p = default_parameters() if p is None else p
    return (p.k_cat_A * (p.k_off_B + p.k_cat_B)) / (
        p.k_cat_B * (p.k_off_A + p.k_cat_A)
    )


def P_A_analytic(p: Parameters | None = None) -> float:
    """``P_A = phi / (1 + phi)``, equivalently ``a_A / (a_A + a_B)``."""
    f = phi(p)
    return f / (1.0 + f)


def P_B_analytic(p: Parameters | None = None) -> float:
    """``P_B = 1 - P_A``."""
    return 1.0 - P_A_analytic(p)


def P_A_from_phi(phi_values):
    """``P_A`` as a function of ``phi`` alone (used for the Fig. 1C curve)."""
    f = np.asarray(phi_values, dtype=float)
    return f / (1.0 + f)


# ==========================================================================
# Equation 2 - timescale to reach A or B
# ==========================================================================

def tau(R, p: Parameters | None = None):
    """Equation 2: characteristic time to commit to A or B, in SECONDS.

    ``tau = (1 + (k_on/k_off_A + k_on/k_off_B)*R)
            / ((k_cat_A*k_on/k_off_A + k_cat_B*k_on/k_off_B)*R)``

    ``k_on/k_off`` has units nM^-1, so the numerator is dimensionless and the
    denominator is s^-1.  Accepts scalars or arrays of ``R`` (nM).

    This is the mean first-passage time out of the unbound/bound cycle in the
    ``k_off >> k_cat`` limit; see :func:`tau_exact_mfpt` for the unapproximated
    expression.
    """
    p = default_parameters() if p is None else p
    R = np.asarray(R, dtype=float)
    ratio_A = p.k_on / p.k_off_A
    ratio_B = p.k_on / p.k_off_B
    numerator = 1.0 + (ratio_A + ratio_B) * R
    denominator = (p.k_cat_A * ratio_A + p.k_cat_B * ratio_B) * R
    with np.errstate(divide="ignore", invalid="ignore"):
        out = numerator / denominator
    return out if out.ndim else float(out)


def tau_limit_infinite_R(p: Parameters | None = None) -> float:
    """``lim_{R->inf} tau``: the binding step becomes instantaneous.

    ``= (k_on/k_off_A + k_on/k_off_B) / (k_cat_A*k_on/k_off_A + k_cat_B*k_on/k_off_B)``
    """
    p = default_parameters() if p is None else p
    ratio_A = p.k_on / p.k_off_A
    ratio_B = p.k_on / p.k_off_B
    return (ratio_A + ratio_B) / (p.k_cat_A * ratio_A + p.k_cat_B * ratio_B)


def tau_exact_mfpt(R, p: Parameters | None = None):
    """Exact mean first-passage time from state 0 to A or B, in seconds.

    Obtained from the embedded Markov chain without the ``k_off >> k_cat``
    simplification.  With ``w_X = k_cat_X/(k_off_X + k_cat_X)`` the probability
    of catalysis per binding event and ``s = k_on*R`` the per-site binding
    rate, the mean exit time from state 0 is

    ``T = (1/(2s) + (1/2)(1/(k_off_A+k_cat_A) + 1/(k_off_B+k_cat_B)))
         / ((w_A + w_B)/2)``

    Equation 2 is recovered by replacing ``k_off_X + k_cat_X`` with
    ``k_off_X``.  Provided so the size of that approximation can be quoted.
    """
    p = default_parameters() if p is None else p
    R = np.asarray(R, dtype=float)
    tot_A = p.k_off_A + p.k_cat_A
    tot_B = p.k_off_B + p.k_cat_B
    w_A = p.k_cat_A / tot_A
    w_B = p.k_cat_B / tot_B
    with np.errstate(divide="ignore", invalid="ignore"):
        time_per_cycle = 1.0 / (2.0 * p.k_on * R) + 0.5 * (1.0 / tot_A + 1.0 / tot_B)
        out = time_per_cycle / (0.5 * (w_A + w_B))
    return out if out.ndim else float(out)


def undifferentiated_fraction(T, R, p: Parameters | None = None):
    """Fraction still in states 0/IA/IB at observation time ``T``.

    Single-exponential approximation ``exp(-T/tau(R))`` with ``tau`` from
    Equation 2.  ``T`` in seconds, ``R`` in nM.
    """
    return np.exp(-np.asarray(T, dtype=float) / tau(R, p))


# ==========================================================================
# Master equation (Supplementary Equations S.1-S.6)
# ==========================================================================

def generator_matrix(R: float, p: Parameters | None = None) -> np.ndarray:
    """5x5 generator ``M`` with ``dy/dt = M @ y`` at fixed recombinase ``R``.

    Every column sums to zero, so total probability is conserved exactly by
    any linear integrator.
    """
    p = default_parameters() if p is None else p
    s = p.k_on * R  # binding rate to one site
    M = np.zeros((5, 5), dtype=float)
    # dp_0  = -2*k_on*R*p_0 + k_off_A*p_IA + k_off_B*p_IB
    M[I_0, I_0] = -2.0 * s
    M[I_0, I_IA] = p.k_off_A
    M[I_0, I_IB] = p.k_off_B
    # dp_IA =  k_on*R*p_0 - (k_off_A + k_cat_A)*p_IA
    M[I_IA, I_0] = s
    M[I_IA, I_IA] = -(p.k_off_A + p.k_cat_A)
    # dp_IB =  k_on*R*p_0 - (k_off_B + k_cat_B)*p_IB
    M[I_IB, I_0] = s
    M[I_IB, I_IB] = -(p.k_off_B + p.k_cat_B)
    # dp_A  = k_cat_A*p_IA   <- fed by the INTERMEDIATE p_IA, not by p_A
    M[I_A, I_IA] = p.k_cat_A
    # dp_B  = k_cat_B*p_IB   <- fed by the INTERMEDIATE p_IB, not by p_B
    M[I_B, I_IB] = p.k_cat_B
    return M


def master_rhs(t: float, y: np.ndarray, R: float,
               p: Parameters | None = None) -> np.ndarray:
    """Right-hand side of the five-state master equation.

    ``t`` is unused (the system is autonomous at fixed ``R``) but kept for the
    ``solve_ivp`` signature.

    Note that the absorbing states A and B appear ONLY on the left-hand sides:
    their inflow terms reference the intermediates ``p_IA`` and ``p_IB``.
    """
    p = default_parameters() if p is None else p
    p_0, p_IA, p_IB = y[I_0], y[I_IA], y[I_IB]
    s = p.k_on * R
    dp_0 = -2.0 * s * p_0 + p.k_off_A * p_IA + p.k_off_B * p_IB
    dp_IA = s * p_0 - (p.k_off_A + p.k_cat_A) * p_IA
    dp_IB = s * p_0 - (p.k_off_B + p.k_cat_B) * p_IB
    dp_A = p.k_cat_A * p_IA   # from the intermediate IA
    dp_B = p.k_cat_B * p_IB   # from the intermediate IB
    return np.array([dp_0, dp_IA, dp_IB, dp_A, dp_B])


def initial_state() -> np.ndarray:
    """``p_0(0) = 1``, every other state 0."""
    y0 = np.zeros(5)
    y0[I_0] = 1.0
    return y0


def solve_master(R: float, t_eval: np.ndarray,
                 p: Parameters | None = None,
                 rtol: float = 1e-10, atol: float = 1e-12):
    """Integrate the master equation at constant ``R``.

    Returns an array of shape ``(len(t_eval), 5)``.
    """
    t_eval = np.asarray(t_eval, dtype=float)
    sol = solve_ivp(
        master_rhs,
        (float(t_eval[0]), float(t_eval[-1])),
        initial_state(),
        t_eval=t_eval,
        args=(R, p),
        method="LSODA",
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"master-equation integration failed: {sol.message}")
    return sol.y.T


# --------------------------------------------------------------------------
# Time-varying R: vectorised integration along Langevin trajectories
# --------------------------------------------------------------------------

#: Order of the Taylor series used by :func:`_expm_batch`.  Combined with
#: scaling so that ||A||_1 <= 1/2, the truncation error is ~(1/2)^13/13! and
#: the exponential is exact to machine precision.
_EXPM_TAYLOR_ORDER = 12
_EXPM_SCALE_TARGET = 0.5


def _expm_batch(A: np.ndarray) -> np.ndarray:
    """Matrix exponential of a stack of matrices, ``A`` of shape ``(M, n, n)``.

    Scaling and squaring with a Taylor series.  Written out rather than taken
    from SciPy because it has to run once per time step for every cell, and
    the batched form is two orders of magnitude faster here.  Validated
    against ``scipy.linalg.expm`` in the test suite.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[-1]

    # 1-norm of each matrix: largest absolute column sum.
    norm = np.abs(A).sum(axis=-2).max(axis=-1)
    with np.errstate(divide="ignore"):
        squarings = np.ceil(np.log2(norm / _EXPM_SCALE_TARGET))
    squarings = np.maximum(0, np.nan_to_num(squarings, neginf=0.0)).astype(np.int64)

    scaled = A / np.ldexp(1.0, squarings)[:, None, None]

    eye = np.broadcast_to(np.eye(n), A.shape)
    result = eye.copy()
    term = eye.copy()
    for k in range(1, _EXPM_TAYLOR_ORDER + 1):
        term = (term @ scaled) / k
        result = result + term

    for level in range(int(squarings.max()) if squarings.size else 0):
        sel = squarings > level
        if sel.any():
            result[sel] = result[sel] @ result[sel]
    return result


def _generator_batch(R: np.ndarray, p: Parameters) -> np.ndarray:
    """Stack of generators ``M(R)``, one per cell; shape ``(M, 5, 5)``.

    Same entries as :func:`generator_matrix`.  The rows for the absorbing
    states A and B are fed by the INTERMEDIATES p_IA and p_IB; neither
    absorbing state appears on any right-hand side.
    """
    s = p.k_on * np.asarray(R, dtype=float)
    M = np.zeros((s.size, 5, 5), dtype=float)
    M[:, I_0, I_0] = -2.0 * s
    M[:, I_0, I_IA] = p.k_off_A
    M[:, I_0, I_IB] = p.k_off_B
    M[:, I_IA, I_0] = s
    M[:, I_IA, I_IA] = -(p.k_off_A + p.k_cat_A)
    M[:, I_IB, I_0] = s
    M[:, I_IB, I_IB] = -(p.k_off_B + p.k_cat_B)
    M[:, I_A, I_IA] = p.k_cat_A   # intermediate IA feeds A
    M[:, I_B, I_IB] = p.k_cat_B   # intermediate IB feeds B
    return M


#: A cell is treated as finished once its total probability of still being in
#: states 0/IA/IB falls below this value.  A and B are absorbing, so such a
#: cell's state vector can no longer change by more than this amount; skipping
#: it is what keeps the noise sweep tractable.
ABSORBED_TOL = 1e-14


def integrate_master_along_R(R_traj: np.ndarray, dt: float,
                             p: Parameters | None = None,
                             return_full: bool = False):
    """Integrate the five-state system along time-varying ``R(t)``.

    Parameters
    ----------
    R_traj
        Shape ``(M, n_steps + 1)``: recombinase concentration for ``M``
        independent cells on a uniform grid of spacing ``dt`` (seconds).
    dt
        Grid spacing in seconds.
    return_full
        Return the whole history instead of only the final state.

    ``R`` is held at its left-endpoint value across each grid interval, which
    is exactly how the Euler-Maruyama trajectory itself is defined.  The
    system is then linear with constant coefficients on that interval, so the
    update ``y <- expm(M(R) dt) y`` is exact; there is no integration error
    from the ODE solver, only the discretisation already present in ``R(t)``.
    Total probability is conserved because every column of ``M`` sums to zero.

    Returns the final probability vector, shape ``(M, 5)``, or the full
    history of shape ``(M, n_steps + 1, 5)``.
    """
    p = default_parameters() if p is None else p
    R_traj = np.atleast_2d(np.asarray(R_traj, dtype=float))
    n_cells, n_points = R_traj.shape

    y = np.zeros((n_cells, 5))
    y[:, I_0] = 1.0

    history = np.empty((n_cells, n_points, 5)) if return_full else None
    if return_full:
        history[:, 0, :] = y

    for i in range(n_points - 1):
        active = np.flatnonzero(y[:, :3].sum(axis=1) > ABSORBED_TOL)
        if active.size == 0:
            if return_full:
                history[:, i + 1:, :] = y[:, None, :]
            break
        propagator = _expm_batch(_generator_batch(R_traj[active, i], p) * dt)
        y[active] = np.einsum("cij,cj->ci", propagator, y[active])
        if return_full:
            history[:, i + 1, :] = y

    return history if return_full else y
