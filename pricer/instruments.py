from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class Instrument(ABC):
    """Un contrat financier : il sait ce qu'il paie, pas ce qu'il vaut."""

    @abstractmethod
    def payoff(self, spot):
        """Ce que rapporte le contrat à l'exercice, pour un ou plusieurs spots."""


@dataclass(frozen=True)
class VanillaOption(Instrument):
    """Clauses et payoff communs aux options européennes et américaines."""

    strike: float           # K
    maturity: float         # T, en années
    option_type: str = "call"

    def __post_init__(self):
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type doit valoir 'call' ou 'put'.")
        if self.strike <= 0 or self.maturity <= 0:
            raise ValueError("Le strike et la maturité doivent être positifs.")

    def payoff(self, spot):
        spot = np.asarray(spot, dtype=float)
        if self.option_type == "call":
            return np.maximum(spot - self.strike, 0.0)
        return np.maximum(self.strike - spot, 0.0)


@dataclass(frozen=True)
class EuropeanOption(VanillaOption):
    """Exerçable uniquement à maturité."""


@dataclass(frozen=True)
class AmericanOption(VanillaOption):
    """Exerçable à tout instant jusqu'à maturité."""