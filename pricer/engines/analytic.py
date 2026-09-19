import numpy as np
from scipy.stats import norm

from pricer.engines.base import PricingEngine
from pricer.instruments import EuropeanOption


class BlackScholesEngine(PricingEngine):
    """Prix exact d'une option européenne dans le modèle de Black-Scholes."""

    def price(self, instrument, market):
        if not isinstance(instrument, EuropeanOption):
            raise TypeError("Black-Scholes ne valorise que les options européennes.")

        S, r, q, sigma = market.spot, market.rate, market.dividend, market.vol
        K, T = instrument.strike, instrument.maturity

        d1, d2 = self.d1_d2(S, K, T, r, q, sigma)
        spot_actualise = S * np.exp(-q * T)     # valeur actuelle de l'action livrée
        strike_actualise = K * np.exp(-r * T)   # valeur actuelle du strike payé

        if instrument.option_type == "call":
            return spot_actualise * norm.cdf(d1) - strike_actualise * norm.cdf(d2)
        return strike_actualise * norm.cdf(-d2) - spot_actualise * norm.cdf(-d1)

    @staticmethod
    def d1_d2(S, K, T, r, q, sigma):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2