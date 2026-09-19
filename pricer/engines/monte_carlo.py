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

    def __init__(self, n_paths=100_000, seed=None, antithetic=False):
        if antithetic and n_paths % 2 != 0:
            raise ValueError("Avec les variables antithétiques, n_paths doit être pair.")
        self.n_paths = n_paths      # nombre total de spots simulés (budget de calcul)
        self.seed = seed            # graine fixée = résultats reproductibles
        self.antithetic = antithetic

    def price(self, instrument, market):
        return self.simulate(instrument, market).price

    def simulate(self, instrument, market):
        if not isinstance(instrument, EuropeanOption):
            raise TypeError("Ce moteur ne gère pour l'instant que les options européennes.")

        rng = np.random.default_rng(self.seed)

        if self.antithetic:
            Z = rng.standard_normal(self.n_paths // 2)
            # Chaque échantillon est la moyenne d'une paire (Z, -Z) : les paires sont indépendantes.
            echantillons = 0.5 * (
                self._payoffs_actualises(instrument, market, Z)
                + self._payoffs_actualises(instrument, market, -Z)
            )
        else:
            Z = rng.standard_normal(self.n_paths)
            echantillons = self._payoffs_actualises(instrument, market, Z)

        prix = echantillons.mean()
        erreur_std = echantillons.std(ddof=1) / np.sqrt(len(echantillons))
        return MCResult(float(prix), float(erreur_std), self.n_paths)

    @staticmethod
    def _payoffs_actualises(instrument, market, Z):
        """Transforme des tirages normaux en payoffs actualisés."""
        S, r, q, sigma = market.spot, market.rate, market.dividend, market.vol
        T = instrument.maturity
        S_T = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        return np.exp(-r * T) * instrument.payoff(S_T)