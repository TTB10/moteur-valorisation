import numpy as np

from pricer.engines.base import PricingEngine
from pricer.instruments import EuropeanOption


class BinomialTreeEngine(PricingEngine):
    """Arbre de Cox-Ross-Rubinstein : remontée par actualisation risque-neutre."""

    def __init__(self, n_steps=500):
        self.n_steps = n_steps

    def price(self, instrument, market):
        if not isinstance(instrument, EuropeanOption):
            raise TypeError("Ce moteur ne gère pour l'instant que les options européennes.")

        S, r, q, sigma = market.spot, market.rate, market.dividend, market.vol
        T, N = instrument.maturity, self.n_steps

        dt = T / N
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        croissance = np.exp((r - q) * dt)
        if not d < croissance < u:
            raise ValueError("Pas de temps trop grand : p sort de [0, 1], augmente n_steps.")
        p = (croissance - d) / (u - d)
        actualisation = np.exp(-r * dt)

        # Spots finaux : j = nombre de hausses, de 0 à N. S * u^j * d^(N-j) = S * exp(sigma*sqrt(dt)*(2j - N))
        j = np.arange(N + 1)
        S_T = S * np.exp(sigma * np.sqrt(dt) * (2 * j - N))
        valeurs = instrument.payoff(S_T)

        # Remontée : le nœud j a pour successeurs j+1 (hausse) et j (baisse).
        for _ in range(N):
            valeurs = actualisation * (p * valeurs[1:] + (1 - p) * valeurs[:-1])

        return float(valeurs[0])