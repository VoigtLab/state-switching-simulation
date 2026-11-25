"""Where the switch becomes slower than cell division.

The switch is useful only while a cell commits faster than it divides.  If
commitment is slower, daughters inherit an uncommitted switch and
undifferentiated cells accumulate.  This module locates that boundary: for a
given division time, the value of each parameter at which the commitment
timescale ``tau`` equals the division time.

Reducing Equation 2
-------------------
Equation 2 as written carries ``k_on`` explicitly::

    tau = (1 + (k_on/k_off_A + k_on/k_off_B)*R)
          / ((k_cat_A*k_on/k_off_A + k_cat_B*k_on/k_off_B)*R)

The model does not treat ``k_off`` as independent: it derives it from the
measured affinity, ``k_off_X = K_D_X * k_on``.  Substituting that gives
``k_on/k_off_X = 1/K_D_X``, and for equal sites (``K_D_A = K_D_B = K_D``,
``k_cat_A = k_cat_B = k_cat``) Equation 2 collapses to

    tau = (K_D/2 + R) / (R * k_cat)                                       (T.1)

**``k_on`` has cancelled.**  ``d(tau)/d(k_on) = 0`` exactly, which
``symbolic_report`` confirms with SymPy.  The commitment timescale is set by
the affinity, the catalytic rate and the recombinase concentration; the
association rate drops out because it scales binding and unbinding equally.
So there is no ``k_on`` threshold in Equation 2, and ``K_D`` is the binding
parameter that matters.

The exact mean first-passage time, which does not assume ``k_off >> k_cat``,
does retain a weak ``k_on`` dependence through the search time ``1/(2*k_on*R)``:

    T = (K_D*k_on + 2*R*k_on + k_cat) / (2*R*k_cat*k_on)                  (T.2)

so a ``k_on`` threshold exists in the exact treatment and is reported for
completeness by :func:`k_on_threshold_exact`.

Thresholds
----------
Setting (T.1) equal to the division time ``t_div`` and solving gives

    R*     = K_D / (2*(k_cat*t_div - 1))        switch is slower below R*
    k_cat* = (K_D/2 + R) / (R*t_div)            slower below k_cat*
    K_D*   = 2*R*(k_cat*t_div - 1)              slower above K_D*

and from (T.2)

    k_on*  = k_cat / (2*R*k_cat*t_div - K_D - 2*R)   slower below k_on*

Each exists only where its denominator is positive.  ``R*`` and ``k_cat*``
require ``k_cat*t_div > 1``: a switch whose catalytic step is slower than one
turnover per division can never keep up at any concentration.
"""

from __future__ import annotations

import numpy as np

from model import tau, tau_exact_mfpt
from parameters import GAMMA, K_CAT_A, K_D, K_ON, R_SS, Parameters, default_parameters

# Division times spanning laboratory culture to host and soil colonisation.
#
# NOTE: these are round numbers chosen to bracket a plausible range, not values
# taken from a specific measurement.  The 77 min entry is the one the model
# itself uses (gamma = 1.5e-4 s^-1).  Anything quoted in the manuscript for the
# lab, mouse or soil endpoints needs a citation supplied by the authors.
DIVISION_TIMES_S = {
    "20 min (lab, rich medium)": 20 * 60.0,
    "40 min (lab, minimal medium)": 40 * 60.0,
    "77 min (this model, 1/gamma ln2)": np.log(2.0) / GAMMA,
    "4 h (slow growth)": 4 * 3600.0,
    "24 h (host / soil colonisation)": 24 * 3600.0,
}

# Continuous range for plotting a threshold against division time.
T_DIV_RANGE_S = (10 * 60.0, 72 * 3600.0)
N_T_DIV = 200


def _equal_site_values(p: Parameters | None = None):
    """Return ``(K_D, k_cat, k_on)``, requiring the two sites to be equal.

    The reduced form (T.1) is only valid when both sites share an affinity and
    a catalytic rate, which is the parameter set of the manuscript.
    """
    p = p or default_parameters()
    if not np.isclose(p.K_D_A, p.K_D_B):
        raise ValueError("reduced tau requires K_D_A == K_D_B")
    if not np.isclose(p.k_cat_A, p.k_cat_B):
        raise ValueError("reduced tau requires k_cat_A == k_cat_B")
    return p.K_D_A, p.k_cat_A, p.k_on


def tau_reduced(R, p: Parameters | None = None):
    """Equation 2 in the reduced form (T.1), as a cross-check on `model.tau`."""
    K_D_value, k_cat, _ = _equal_site_values(p)
    R = np.asarray(R, dtype=float)
    return (K_D_value / 2.0 + R) / (R * k_cat)


def doubling_time_from_gamma(gamma: float = GAMMA) -> float:
    """Division time implied by a dilution rate, ``ln(2)/gamma`` in seconds."""
    return float(np.log(2.0) / gamma)


# ==========================================================================
# Thresholds
# ==========================================================================

def R_threshold(t_div, p: Parameters | None = None):
    """Recombinase concentration (nM) at which ``tau`` equals ``t_div``.

    Below this concentration the switch is slower than division.  Returns NaN
    where ``k_cat*t_div <= 1``, i.e. where no concentration is fast enough.
    """
    K_D_value, k_cat, _ = _equal_site_values(p)
    t_div = np.asarray(t_div, dtype=float)
    denominator = 2.0 * (k_cat * t_div - 1.0)
    return np.where(denominator > 0.0, K_D_value / denominator, np.nan)


def k_cat_threshold(t_div, R: float = R_SS, p: Parameters | None = None):
    """Catalytic rate (s^-1) at which ``tau`` equals ``t_div`` at fixed ``R``.

    Below this rate the switch is slower than division.
    """
    K_D_value, _, _ = _equal_site_values(p)
    t_div = np.asarray(t_div, dtype=float)
    return (K_D_value / 2.0 + R) / (R * t_div)


def K_D_threshold(t_div, R: float = R_SS, p: Parameters | None = None):
    """Affinity (nM) at which ``tau`` equals ``t_div`` at fixed ``R``.

    *Above* this affinity — that is, for weaker binding — the switch is slower
    than division.  Returns NaN where ``k_cat*t_div <= 1``.
    """
    _, k_cat, _ = _equal_site_values(p)
    t_div = np.asarray(t_div, dtype=float)
    threshold = 2.0 * R * (k_cat * t_div - 1.0)
    return np.where(threshold > 0.0, threshold, np.nan)


def k_on_threshold_exact(t_div, R: float = R_SS, p: Parameters | None = None):
    """Association rate (nM^-1 s^-1) at which the *exact* MFPT equals ``t_div``.

    Equation 2 has no ``k_on`` threshold because ``k_on`` cancels; this uses
    the unapproximated expression (T.2), where it survives in the search term.
    Below this rate the switch is slower than division.  Returns NaN where no
    positive solution exists.
    """
    K_D_value, k_cat, _ = _equal_site_values(p)
    t_div = np.asarray(t_div, dtype=float)
    denominator = 2.0 * R * k_cat * t_div - K_D_value - 2.0 * R
    return np.where(denominator > 0.0, k_cat / denominator, np.nan)


def threshold_table(division_times=None, R: float = R_SS,
                    p: Parameters | None = None):
    """One row per division time, with each threshold and its safety margin.

    The margin is the factor between the value the model uses and the
    threshold: how far the operating point sits from the boundary.
    """
    division_times = division_times or DIVISION_TIMES_S
    K_D_value, k_cat, k_on_value = _equal_site_values(p)
    rows = []
    for label, t_div in division_times.items():
        R_star = float(R_threshold(t_div, p))
        k_cat_star = float(k_cat_threshold(t_div, R, p))
        K_D_star = float(K_D_threshold(t_div, R, p))
        k_on_star = float(k_on_threshold_exact(t_div, R, p))
        rows.append({
            "label": label,
            "t_div_s": float(t_div),
            "t_div_min": float(t_div) / 60.0,
            "tau_at_R_ss_s": float(tau(R, p)),
            "R_threshold_nM": R_star,
            "R_margin": R / R_star if R_star > 0 else np.nan,
            "k_cat_threshold_per_s": k_cat_star,
            "k_cat_margin": k_cat / k_cat_star if k_cat_star > 0 else np.nan,
            "K_D_threshold_nM": K_D_star,
            "K_D_margin": K_D_star / K_D_value if K_D_star > 0 else np.nan,
            "k_on_threshold_exact": k_on_star,
            "k_on_margin": k_on_value / k_on_star if k_on_star > 0 else np.nan,
        })
    return rows


def threshold_curves(t_div_range=T_DIV_RANGE_S, n=N_T_DIV,
                     R: float = R_SS, p: Parameters | None = None):
    """Each threshold against division time, for plotting."""
    t_div = np.logspace(np.log10(t_div_range[0]), np.log10(t_div_range[1]), n)
    return {
        "t_div_s": t_div,
        "t_div_min": t_div / 60.0,
        "R_threshold_nM": R_threshold(t_div, p),
        "k_cat_threshold_per_s": k_cat_threshold(t_div, R, p),
        "K_D_threshold_nM": K_D_threshold(t_div, R, p),
        "k_on_threshold_exact": k_on_threshold_exact(t_div, R, p),
    }


# ==========================================================================
# Symbolic confirmation
# ==========================================================================

def symbolic_report() -> dict:
    """Confirm the reduction and the thresholds with SymPy.

    Returns the reduced ``tau``, the derivative with respect to ``k_on`` (which
    is zero), and each solved threshold, all as strings.
    """
    import sympy as sp

    k_on_s, K_D_s, k_cat_s, R_s, t_div_s = sp.symbols(
        "k_on K_D k_cat R t_div", positive=True
    )
    k_off_s = K_D_s * k_on_s                 # the model derives k_off from K_D

    tau_s = sp.simplify(
        (1 + 2 * (k_on_s / k_off_s) * R_s)
        / (2 * (k_cat_s * k_on_s / k_off_s) * R_s)
    )
    w = k_cat_s / (k_off_s + k_cat_s)
    mfpt_s = sp.together(sp.simplify(
        (1 / (2 * k_on_s * R_s) + 1 / (k_off_s + k_cat_s)) / w
    ))

    def _solve(expression, symbol):
        solutions = sp.solve(sp.Eq(expression, t_div_s), symbol)
        return str(solutions[0]) if solutions else "no solution"

    return {
        "tau_reduced": str(tau_s),
        "dtau_dk_on": str(sp.simplify(sp.diff(tau_s, k_on_s))),
        "dtau_dK_D": str(sp.simplify(sp.diff(tau_s, K_D_s))),
        "mfpt_exact": str(mfpt_s),
        "dmfpt_dk_on": str(sp.simplify(sp.diff(mfpt_s, k_on_s))),
        "R_threshold": _solve(tau_s, R_s),
        "k_cat_threshold": _solve(tau_s, k_cat_s),
        "K_D_threshold": _solve(tau_s, K_D_s),
        "k_on_threshold_from_Eq2": _solve(tau_s, k_on_s),
        "k_on_threshold_from_mfpt": _solve(mfpt_s, k_on_s),
    }


# ==========================================================================
# Dimensionless form
# ==========================================================================
#
# Dividing (T.1) through by 1/k_cat leaves a single dimensionless statement:
#
#     tau * k_cat = 1 + K_D/(2*R)                                      (T.3)
#
# so the switch is faster than division exactly when
#
#     k_cat * t_div  >  1 + K_D/(2*R)                                  (T.4)
#
# Two dimensionless groups govern the whole boundary: the catalytic turnovers
# available per division, ``k_cat*t_div``, and the occupancy term ``K_D/(2*R)``.
# Individual values of k_on, k_off, k_cat, K_D and R never need to be quoted
# separately -- only these two combinations, and k_on does not enter either.

K_D_OVER_R_RANGE = (1e-3, 1e3)
N_DIMENSIONLESS = 200


def dimensionless_boundary(K_D_over_R_range=K_D_OVER_R_RANGE,
                           n=N_DIMENSIONLESS):
    """The boundary (T.4) as turnovers per division against ``K_D/R``.

    Returns ``K_D/R`` and the minimum ``k_cat*t_div`` that keeps the switch
    ahead of division at that ratio.  Points above the curve commit faster
    than the cell divides.
    """
    ratio = np.logspace(np.log10(K_D_over_R_range[0]),
                        np.log10(K_D_over_R_range[1]), n)
    return ratio, 1.0 + ratio / 2.0


def operating_point(t_div: float, R: float = R_SS,
                    p: Parameters | None = None) -> dict:
    """Where the manuscript's parameters sit relative to the boundary (T.4)."""
    K_D_value, k_cat, _ = _equal_site_values(p)
    ratio = K_D_value / R
    required = 1.0 + ratio / 2.0
    available = k_cat * t_div
    return {
        "K_D_over_R": ratio,
        "turnovers_required": required,
        "turnovers_available": available,
        "margin": available / required,
        "faster_than_division": bool(available > required),
    }
