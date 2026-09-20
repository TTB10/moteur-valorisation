import numpy as np
from scipy.stats import norm

from pricer.engines.base import PricingEngine
from pricer.greeks import Greeks
from pricer.instruments import EuropeanOption


class BlackScholesEngine(PricingEngine):
    """Prix et grecques exacts d'une option européenne dans le modèle de Black-Scholes."""

    def price(self, instrument, market):
        S, K, T, r, q, sigma, d1, d2 = self._parametres(instrument, market)
        spot_actualise = S * np.exp(-q * T)     # valeur actuelle de l'action livrée
        strike_actualise = K * np.exp(-r * T)   # valeur actuelle du strike payé

        if instrument.option_type == "call":
            return spot_actualise * norm.cdf(d1) - strike_actualise * norm.cdf(d2)
        return strike_actualise * norm.cdf(-d2) - spot_actualise * norm.cdf(-d1)

    def greeks(self, instrument, market):
        """Les cinq grecques par formule analytique."""
        S, K, T, r, q, sigma, d1, d2 = self._parametres(instrument, market)
        est_call = instrument.option_type == "call"
        racine_T = np.sqrt(T)
        esc_q = np.exp(-q * T)      # escompte du dividende
        esc_r = np.exp(-r * T)      # actualisation au taux sans risque
        densite = norm.pdf(d1)      # phi(d1)

        delta = esc_q * norm.cdf(d1) if est_call else -esc_q * norm.cdf(-d1)
        gamma = esc_q * densite / (S * sigma * racine_T)
        vega = S * esc_q * densite * racine_T

        # Theta = -S e^-qT phi(d1) sigma / (2 sqrt(T))  +/- termes de portage
        erosion = -S * esc_q * densite * sigma / (2 * racine_T)
        if est_call:
            theta = erosion + q * S * esc_q * norm.cdf(d1) - r * K * esc_r * norm.cdf(d2)
            rho = K * T * esc_r * norm.cdf(d2)
        else:
            theta = erosion - q * S * esc_q * norm.cdf(-d1) + r * K * esc_r * norm.cdf(-d2)
            rho = -K * T * esc_r * norm.cdf(-d2)

        return Greeks(float(delta), float(gamma), float(vega), float(theta), float(rho))

    def _parametres(self, instrument, market):
        """Contrôle le type d'instrument et renvoie les paramètres, d1 et d2."""
        if not isinstance(instrument, EuropeanOption):
            raise TypeError("Black-Scholes ne valorise que les options européennes.")
        S, r, q, sigma = market.spot, market.rate, market.dividend, market.vol
        K, T = instrument.strike, instrument.maturity
        d1, d2 = self.d1_d2(S, K, T, r, q, sigma)
        return S, K, T, r, q, sigma, d1, d2

    @staticmethod
    def d1_d2(S, K, T, r, q, sigma):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return d1, d1 - sigma * np.sqrt(T)