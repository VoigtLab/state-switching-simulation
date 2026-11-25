"""Single source of truth for every model parameter.

All other modules import from here; no numerical literal defined in this file
is repeated anywhere else in the repository.

Units
-----
concentration : nM
time          : seconds
rates         : s^-1 (bimolecular: nM^-1 s^-1)

Recombinase abundance is expressed in nM throughout.  For E. coli, one
molecule per cell corresponds to approximately 1 nM (BioNumbers BNID 102068;
Milo & Phillips, *Cell Biology by the Numbers*), so molecule counts and
nanomolar concentrations are interchangeable here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

# --------------------------------------------------------------------------
# Kinetic parameters of the five-state recombinase switch
# --------------------------------------------------------------------------

#: Recombinase association rate to either att site.
#: Literature estimate for a protein locating a binding site in a crowded
#: cytoplasm without facilitated diffusion.
K_ON = 0.001  # nM^-1 s^-1

#: Dissociation constant of Bxb1 for its att site (Ghosh et al. 2005;
#: Singh et al. 2013).
K_D = 70.0  # nM

#: Dissociation rates, k_off = K_D * k_on.  Equal for both sites here;
#: they are the parameters that change when a different att variant is used.
K_OFF_A = K_D * K_ON  # s^-1  -> 0.07
K_OFF_B = K_D * K_ON  # s^-1  -> 0.07

#: Catalytic (strand-exchange) rates.  Chao et al. 2021 FEBS J, Table 1
#: reports 3.69e-3 s^-1; rounded to 0.004 s^-1 and taken equal for the
#: inversion (A) and excision (B) paths.
K_CAT_A = 0.004  # s^-1
K_CAT_B = 0.004  # s^-1

# --------------------------------------------------------------------------
# Unequal catalytic rates
# --------------------------------------------------------------------------
# A biased example, used to exercise the asymmetric case.  The switch above has
# k_cat_A = k_cat_B, so phi = 1 and the two fates are equally likely; with
# unequal catalytic rates phi departs from 1 and the population splits in a
# fixed, tunable ratio instead.  These values are not those of the switch
# characterised here.
K_CAT_A_UNEQUAL = 0.023  # s^-1
K_CAT_B_UNEQUAL = 0.004  # s^-1

# --------------------------------------------------------------------------
# Recombinase expression (Equation 3, Langevin)
# --------------------------------------------------------------------------

#: Zeroth-order production of recombinase.  Corresponds to a transcriptional
#: burst size of about 7 proteins per mRNA at ~5 mRNA per cell cycle, between
#: the measured median (3) and mean (21) for E. coli (Taniguchi et al.,
#: Science 2010).  See the README for the full derivation.
BETA_0 = 0.0075   # nM s^-1, zeroth-order production
GAMMA = 0.00015   # s^-1, first-order dilution/degradation (77 min doubling)
R_SS = BETA_0 / GAMMA  # nM, deterministic steady state -> 50.0

#: Initial recombinase concentration for the Langevin trajectories.  Expression
#: starts from zero and accumulates towards R_SS; at t_end the deterministic
#: solution has reached 44.23 nM, not R_SS.
R_INITIAL = 0.0   # nM

#: Noise amplitude of the multiplicative Langevin term.  Two conditions are
#: shown in Fig. 1E upper: the deterministic limit and the measured amplitude.
SIGMA_NOISELESS = 0.0    # s^-1/2
SIGMA_NOISY = 0.064      # s^-1/2

#: Noise amplitudes swept in Fig. 1E lower: 16 points at a spacing of 0.005.
#: SIGMA_NOISY is a separate condition for the Fig. 1E upper trace and is
#: deliberately not on this grid.
SIGMA_SWEEP = np.arange(0.0, 0.0751, 0.005)

# --------------------------------------------------------------------------
# Integration grid
# --------------------------------------------------------------------------

DT = 2.0          # s, Euler-Maruyama step
T_END = 14400.0   # s (240 min)

#: Seed for numpy.random.default_rng.  Every stochastic result in the
#: repository is reproducible from this value.
SEED = 2025


@dataclass(frozen=True)
class Parameters:
    """Immutable bundle of the switch kinetics.

    Use :func:`default_parameters` for the manuscript values and
    :meth:`with_` to make a modified copy for sensitivity sweeps.
    """

    k_on: float
    k_off_A: float
    k_off_B: float
    k_cat_A: float
    k_cat_B: float

    def with_(self, **changes: float) -> "Parameters":
        """Return a copy with the named fields replaced."""
        return replace(self, **changes)

    @property
    def K_D_A(self) -> float:
        """Dissociation constant of site A, K_D = k_off / k_on (nM)."""
        return self.k_off_A / self.k_on

    @property
    def K_D_B(self) -> float:
        """Dissociation constant of site B, K_D = k_off / k_on (nM)."""
        return self.k_off_B / self.k_on


def default_parameters() -> Parameters:
    """Kinetic parameters used throughout the manuscript."""
    return Parameters(
        k_on=K_ON,
        k_off_A=K_OFF_A,
        k_off_B=K_OFF_B,
        k_cat_A=K_CAT_A,
        k_cat_B=K_CAT_B,
    )


def unequal_rate_parameters() -> Parameters:
    """A biased switch: unequal catalytic rates, so ``phi != 1``.

    The equal-rate case cannot test the asymmetric branch of Equation 1,
    because every ratio there collapses to 1.  This set gives
    ``phi = 4.575...`` and ``P_A = 0.8206...``, which the workbook checks
    analytically and against the stochastic simulation.
    """
    return default_parameters().with_(
        k_cat_A=K_CAT_A_UNEQUAL,
        k_cat_B=K_CAT_B_UNEQUAL,
    )


def from_KD(k_on: float, K_D_A: float, K_D_B: float,
            k_cat_A: float, k_cat_B: float) -> Parameters:
    """Build :class:`Parameters` from dissociation constants (k_off = K_D*k_on)."""
    return Parameters(
        k_on=k_on,
        k_off_A=K_D_A * k_on,
        k_off_B=K_D_B * k_on,
        k_cat_A=k_cat_A,
        k_cat_B=k_cat_B,
    )
