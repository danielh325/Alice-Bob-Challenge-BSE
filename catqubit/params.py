"""Physical parameters, in the challenge's units: time in us, rates in rad/us."""

import math
from dataclasses import dataclass, replace

# Réglade, Bocquet et al., arXiv:2307.06617, Table S2 (ordinary frequencies, MHz).
G2_MHZ = 0.763
KAPPA_B_MHZ = 2.6
KAPPA_A_MHZ = 9.3e-3
TARGET_ALPHA = 2.1


def to_angular(freq_mhz: float) -> float:
    """Convert an ordinary frequency f (MHz) to an angular rate 2*pi*f (rad/us)."""
    return 2 * math.pi * freq_mhz


@dataclass(frozen=True)
class CatParams:
    """Two-mode cat-qubit parameters. All rates are angular, in rad/us."""

    g2: float
    kappa_b: float
    kappa_a: float
    alpha: float

    def __post_init__(self) -> None:
        if self.g2 <= 0:
            raise ValueError(f"g2 must be positive, got {self.g2}")
        if self.kappa_b <= 0:
            raise ValueError(f"kappa_b must be positive, got {self.kappa_b}")
        if self.kappa_a < 0:
            raise ValueError(f"kappa_a must be non-negative, got {self.kappa_a}")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")

    @classmethod
    def reglade(cls) -> "CatParams":
        """The provided experimental parameters, converted to rad/us."""
        return cls(
            g2=to_angular(G2_MHZ),
            kappa_b=to_angular(KAPPA_B_MHZ),
            kappa_a=to_angular(KAPPA_A_MHZ),
            alpha=TARGET_ALPHA,
        )

    @property
    def eps_d(self) -> float:
        """Buffer drive amplitude that stabilizes alpha^2 = eps_d / g2."""
        return self.alpha**2 * self.g2

    @property
    def kappa2(self) -> float:
        """Effective two-photon dissipation rate after adiabatic elimination."""
        return 4 * self.g2**2 / self.kappa_b

    def adiabatic_ratios(self) -> tuple[float, float]:
        """(g2 / kappa_b, eps_d / kappa_b); both must be << 1 for the effective model."""
        return self.g2 / self.kappa_b, self.eps_d / self.kappa_b

    def with_alpha(self, alpha: float) -> "CatParams":
        """Copy with a new cat amplitude (and hence a new buffer drive)."""
        return replace(self, alpha=alpha)
