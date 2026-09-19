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

    def __init__(self, n_paths=100_000, seed=None, antithetic=False, control_variate=False):
        if antithetic and n_paths % 2 != 0:
            raise ValueError("Avec les variables antithétiques, n_paths doit être pair.")
        self.n_paths = n_paths              # nombre total de spots simulés (budget de calcul)
        self.seed = seed                    # graine fixée = résultats reproductibles
        self.antithetic = antithetic
        self.control_variate = control_variate

    def price(self, instrument, market):
        return self.simulate(instrument, market).price

    def simulate(self, instrument, market):
        if not isinstance(instrument, EuropeanOption):
            raise TypeError("Ce moteur ne gère pour l'instant que les options européennes.")

        rng = np.random.default_rng(self.seed)

        # Y : payoffs actualisés. X : spots finaux actualisés (variable de contrôle).
        if self.antithetic:
            Z = rng.standard_normal(self.n_paths // 2)
            y_plus, x_plus = self._tirer(instrument, market, Z)
            y_moins, x_moins = self._tirer(instrument, market, -Z)
            Y = 0.5 * (y_plus + y_moins)
            X = 0.5 * (x_plus + x_moins)
        else:
            Z = rng.standard_normal(self.n_paths)
            Y, X = self._tirer(instrument, market, Z)

        if self.control_variate:
            esperance_X = market.spot * np.exp(-market.dividend * instrument.maturity)
            beta = np.cov(Y, X)[0, 1] / np.var(X, ddof=1)
            Y = Y - beta * (X - esperance_X)

        prix = Y.mean()
        erreur_std = Y.std(ddof=1) / np.sqrt(len(Y))
        return MCResult(float(prix), float(erreur_std), self.n_paths)

    @staticmethod
    def _tirer(instrument, market, Z):
        """Transforme des tirages normaux en payoffs actualisés et spots finaux actualisés."""
        S, r, q, sigma = market.spot, market.rate, market.dividend, market.vol
        T = instrument.maturity
        S_T = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        actualisation = np.exp(-r * T)
        return actualisation * instrument.payoff(S_T), actualisation * S_T