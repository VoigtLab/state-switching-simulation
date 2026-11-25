"""Sensitivity analysis of the outcome ratio and of the commitment timescale.

Four questions, one per sheet of the workbook (``SensA`` to ``SensD``):

1. Is ``phi`` really independent of the recombinase concentration ``R``, and is
   it invariant under a common rescaling of all four rate constants?
2. Is ``phi`` flat at 1 over realistic ranges of ``k_on`` and ``K_D`` when the
   two catalytic rates are equal?
3. How many cells are still undifferentiated at a fixed observation time when
   ``R`` is low?
4. Among the cells that *have* differentiated, does the A:B ratio depend on
   ``R``?

Questions 3 and 4 together are the boundary condition on "expression-level
independence": low ``R`` changes how many cells commit, not the ratio among
those that do.
"""

from __future__ import annotations

import numpy as np

from model import P_A_analytic, phi, solve_master, tau, undifferentiated_fraction
from parameters import Parameters, default_parameters, from_KD

#: Observation times for the undifferentiated-fraction analysis, in seconds.
OBSERVATION_TIMES_S = {"6 h": 6 * 3600.0, "24 h": 24 * 3600.0}

#: Recombinase concentrations tabulated in the sensitivity analysis.
R_TABLE_NM = (0.05, 0.1, 0.3, 1.0, 50.0)


# ==========================================================================
# 1. Expression-level invariance
# ==========================================================================

def phi_versus_R(R_values, p: Parameters | None = None) -> np.ndarray:
    """``phi`` evaluated at each ``R``.

    ``phi`` has no ``R`` argument, so this recomputes the same expression once
    per concentration: the point is that the numbers come out bit-for-bit
    identical, not merely similar.
    """
    p = default_parameters() if p is None else p
    value = phi(p)
    return np.full(np.shape(R_values), value, dtype=float)


def phi_under_rate_rescaling(scale_values, p: Parameters | None = None) -> np.ndarray:
    """``phi`` when all four rate constants are multiplied by a common factor.

    Numerator and denominator of Equation 1 are both quadratic in the rates,
    so the factor cancels.  This is the kinetic-compensation argument: a
    perturbation (temperature, growth rate) that scales every rate equally
    leaves the fate ratio untouched.
    """
    p = default_parameters() if p is None else p
    out = []
    for c in np.atleast_1d(np.asarray(scale_values, dtype=float)):
        scaled = p.with_(
            k_off_A=p.k_off_A * c,
            k_off_B=p.k_off_B * c,
            k_cat_A=p.k_cat_A * c,
            k_cat_B=p.k_cat_B * c,
        )
        out.append(phi(scaled))
    return np.asarray(out)


def symbolic_invariance_report() -> dict:
    """Prove with sympy that ``phi`` is free of ``R`` and of a common factor.

    Returns a dict of simplified sympy expressions:

    ``phi``             Equation 1 in symbols
    ``phi_rescaled``    Equation 1 after ``k -> c*k`` for all four rates
    ``difference``      ``phi_rescaled - phi``, which simplifies to 0
    ``dphi_dR``         derivative of ``phi`` with respect to ``R``, which is 0
    ``P_A_ode``         steady-state ``p_A`` of the master equation, solved
                        symbolically and simplified; equals ``phi/(1+phi)``
    """
    import sympy as sp

    k_on, R, c = sp.symbols("k_on R c", positive=True)
    k_off_A, k_off_B = sp.symbols("k_off_A k_off_B", positive=True)
    k_cat_A, k_cat_B = sp.symbols("k_cat_A k_cat_B", positive=True)

    expr = (k_cat_A * (k_off_B + k_cat_B)) / (k_cat_B * (k_off_A + k_cat_A))
    rescaled = expr.subs(
        {
            k_off_A: c * k_off_A,
            k_off_B: c * k_off_B,
            k_cat_A: c * k_cat_A,
            k_cat_B: c * k_cat_B,
        }
    )

    # Steady state of the embedded chain, derived rather than assumed:
    # from state 0 the two sites are entered at the same rate k_on*R, so the
    # absorption probabilities depend only on a_X = k_cat_X/(k_off_X+k_cat_X).
    a_A = k_cat_A / (k_off_A + k_cat_A)
    a_B = k_cat_B / (k_off_B + k_cat_B)
    P_A_ode = sp.simplify(a_A / (a_A + a_B))

    return {
        "phi": sp.simplify(expr),
        "phi_rescaled": sp.simplify(rescaled),
        "difference": sp.simplify(rescaled - expr),
        "dphi_dR": sp.simplify(sp.diff(expr, R)),
        "P_A_ode": P_A_ode,
        "P_A_from_phi": sp.simplify(expr / (1 + expr)),
        "P_A_agrees": sp.simplify(P_A_ode - expr / (1 + expr)),
    }


# ==========================================================================
# 2. Parameter grid over k_on and K_D
# ==========================================================================

def phi_grid(k_on_values, K_D_values, k_cat: float | None = None) -> np.ndarray:
    """``phi`` over a ``k_on`` x ``K_D`` grid with ``k_cat_A = k_cat_B``.

    Both sites share the same ``K_D`` and the same ``k_cat``, which is the
    configuration of the switch characterised here; the grid asks whether
    anything about the binding kinetics can move the ratio away from 1.

    Returns an array of shape ``(len(K_D_values), len(k_on_values))`` so that
    it can be passed straight to ``pcolormesh`` with ``K_D`` on the y-axis.
    """
    base = default_parameters()
    k_cat = base.k_cat_A if k_cat is None else k_cat
    k_on_values = np.asarray(k_on_values, dtype=float)
    K_D_values = np.asarray(K_D_values, dtype=float)

    grid = np.empty((K_D_values.size, k_on_values.size), dtype=float)
    for i, K_D in enumerate(K_D_values):
        for j, k_on in enumerate(k_on_values):
            grid[i, j] = phi(from_KD(k_on, K_D, K_D, k_cat, k_cat))
    return grid


# ==========================================================================
# 3. Undifferentiated fraction at low R
# ==========================================================================

def undifferentiated_curve(R_values, T_seconds: float,
                           p: Parameters | None = None) -> np.ndarray:
    """``exp(-T/tau(R))`` over a range of ``R``, using Equation 2."""
    return undifferentiated_fraction(T_seconds, np.asarray(R_values, float), p)


def undifferentiated_exact(R_values, T_seconds: float,
                           p: Parameters | None = None) -> np.ndarray:
    """Undifferentiated fraction from the master equation itself.

    ``p_0 + p_IA + p_IB`` at time ``T``, integrated numerically.  Reported
    alongside the single-exponential approximation so the size of that
    approximation is visible rather than assumed.
    """
    R_values = np.atleast_1d(np.asarray(R_values, dtype=float))
    out = np.empty(R_values.size, dtype=float)
    t_eval = np.array([0.0, float(T_seconds)])
    for i, R in enumerate(R_values):
        y = solve_master(float(R), t_eval, p)[-1]
        out[i] = y[0] + y[1] + y[2]
    return out


def undifferentiated_table(R_values=R_TABLE_NM, p: Parameters | None = None):
    """Rows of ``(R, tau_seconds, tau_hours, undiff@6h, undiff@24h)``."""
    rows = []
    for R in R_values:
        tau_s = float(tau(float(R), p))
        rows.append(
            {
                "R_nM": float(R),
                "tau_s": tau_s,
                "tau_h": tau_s / 3600.0,
                "undiff_6h": float(undifferentiated_fraction(OBSERVATION_TIMES_S["6 h"], R, p)),
                "undiff_24h": float(undifferentiated_fraction(OBSERVATION_TIMES_S["24 h"], R, p)),
            }
        )
    return rows


# ==========================================================================
# 4. Ratio among differentiated cells
# ==========================================================================

def ratio_among_differentiated(R_values, T_seconds: float,
                               p: Parameters | None = None) -> np.ndarray:
    """``p_A / (p_A + p_B)`` at time ``T``, from the master equation.

    Computed from the transient solution, not from Equation 1, so that the
    flatness of the curve is a result of the integration rather than an
    identity built into the code.
    """
    R_values = np.atleast_1d(np.asarray(R_values, dtype=float))
    out = np.empty(R_values.size, dtype=float)
    t_eval = np.array([0.0, float(T_seconds)])
    for i, R in enumerate(R_values):
        y = solve_master(float(R), t_eval, p)[-1]
        p_A, p_B = y[3], y[4]
        out[i] = p_A / (p_A + p_B) if (p_A + p_B) > 0 else np.nan
    return out


def analytic_ratio(p: Parameters | None = None) -> float:
    """The ``R``-independent value the ``SensD`` ratio must lie on."""
    return P_A_analytic(p)
