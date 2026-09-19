import numpy as np

from pricer.engines.base import PricingEngine
from pricer.instruments import AmericanOption, EuropeanOption


class BinomialTreeEngine(PricingEngine):
    """Arbre de Cox-Ross-Rubinstein : remontée par actualisation risque-neutre."""

    def __init__(self, n_steps=500):
        self.n_steps = n_steps

    def price(self, instrument, market):
        if not isinstance(instrument, (EuropeanOption, AmericanOption)):
            raise TypeError("Ce moteur valorise les options européennes et américaines.")

        S, r, q, sigma = market.spot, market.rate, market.dividend, market.vol
        T, N = instrument.maturity, self.n_steps

        dt = T / N
        pas = sigma * np.sqrt(dt)            # u = e^pas, d = e^-pas
        u, d = np.exp(pas), np.exp(-pas)
        croissance = np.exp((r - q) * dt)
        if not d < croissance < u:
            raise ValueError("Pas de temps trop grand : p sort de [0, 1], augmente n_steps.")
        p = (croissance - d) / (u - d)
        actualisation = np.exp(-r * dt)
        americaine = isinstance(instrument, AmericanOption)

        # Nœuds finaux : j = nombre de hausses, spot = S * exp(pas * (2j - N)).
        j = np.arange(N + 1)
        valeurs = instrument.payoff(S * np.exp(pas * (2 * j - N)))

        # Remontée de la date N-1 à la date 0. Le nœud j a pour successeurs j+1 (hausse) et j (baisse).
        for i in range(N - 1, -1, -1):
            valeurs = actualisation * (p * valeurs[1:] + (1 - p) * valeurs[:-1])
            if americaine:
                j = np.arange(i + 1)
                exercice = instrument.payoff(S * np.exp(pas * (2 * j - i)))
                valeurs = np.maximum(valeurs, exercice)   # garder ou exercer : on prend le meilleur

        return float(valeurs[0])