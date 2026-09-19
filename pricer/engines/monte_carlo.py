from dataclasses import dataclass

import numpy as np

from pricer.engines.base import PricingEngine
from pricer.instruments import EuropeanOption


@dataclass(frozen=True)
class MCResult:
    """Résultat d'une simulation : le prix estimé et sa précision."""

    price: float
    std_error: float
    n_paths: int

    @property
    def ci_95(self):
        return (self.price - 1.96 * self.std_error, self.price + 1.96 * self.std_error)


class MonteCarloEngine(PricingEngine):
    """Prix estimé par la moyenne des payoffs actualisés sous la probabilité risque-neutre."""

    def __init__(self, n_paths=100_000, seed=None):
        self.n_paths = n_paths
        self.seed = seed  # graine fixée = résultats reproductibles

    def price(self, instrument, market):
        return self.simulate(instrument, market).price

    def simulate(self, instrument, market):
        if not isinstance(instrument, EuropeanOption):
            raise TypeError("Ce moteur ne gère pour l'instant que les options européennes.")

        S, r, q, sigma = market.spot, market.rate, market.dividend, market.vol
        T = instrument.maturity

        rng = np.random.default_rng(self.seed)
        Z = rng.standard_normal(self.n_paths)
        S_T = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

        payoffs_actualises = np.exp(-r * T) * instrument.payoff(S_T)
        prix = payoffs_actualises.mean()
        erreur_std = payoffs_actualises.std(ddof=1) / np.sqrt(self.n_paths)

        return MCResult(float(prix), float(erreur_std), self.n_paths)