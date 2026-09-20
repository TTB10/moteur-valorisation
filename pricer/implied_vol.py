"""Inversion de Black-Scholes : volatilité implicite par Newton-Raphson avec repli dichotomique."""

from dataclasses import replace

import numpy as np

from pricer.engines.analytic import BlackScholesEngine
from pricer.instruments import EuropeanOption

MOTEUR = BlackScholesEngine()

VOL_MIN = 1e-4      # 0,01 %
VOL_MAX = 5.0       # 500 %, au-delà le prix est indiscernable de sa borne
VEGA_MIN = 1e-6     # en dessous, la volatilité n'est plus identifiable à partir du prix

class NoImpliedVolatility(Exception):
    """Aucune volatilité ne reproduit ce prix : il viole les bornes d'arbitrage."""


def price_bounds(option, market):
    """Bornes du prix, atteintes quand sigma tend vers 0 et vers l'infini."""
    S, K, T = market.spot, option.strike, option.maturity
    forward_actualise = S * np.exp(-market.dividend * T)
    strike_actualise = K * np.exp(-market.rate * T)

    if option.option_type == "call":
        return max(forward_actualise - strike_actualise, 0.0), forward_actualise
    return max(strike_actualise - forward_actualise, 0.0), strike_actualise


def implied_volatility(prix_marche, option, market, tol=1e-8, max_iter=100):
    """Volatilité implicite. Newton tant qu'il reste dans l'encadrement, dichotomie sinon.

    Lève NoImpliedVolatility si le prix viole les bornes d'arbitrage, ou si le vega
    est trop faible pour que la volatilité soit identifiable (option très dans la monnaie).
    """
    if not isinstance(option, EuropeanOption):
        raise TypeError("L'inversion n'est définie que pour les options européennes.")

    borne_basse, borne_haute = price_bounds(option, market)
    if not (borne_basse - 1e-12 < prix_marche < borne_haute + 1e-12):
        raise NoImpliedVolatility(
            f"Prix {prix_marche:.6f} hors des bornes [{borne_basse:.6f}, {borne_haute:.6f}]."
        )

    ecart = lambda sigma: MOTEUR.price(option, replace(market, vol=sigma)) - prix_marche
    vega_de = lambda sigma: MOTEUR.greeks(option, replace(market, vol=sigma)).vega

    # Encadrement initial : l'écart est croissant en sigma.
    bas, haut = VOL_MIN, VOL_MAX
    if ecart(bas) > 0 or ecart(haut) < 0:
        raise NoImpliedVolatility(
            "Prix indiscernable d'une borne : la volatilité n'est pas identifiable."
        )

    sigma = _point_de_depart(prix_marche, option, market)

    for _ in range(max_iter):
        f = ecart(sigma)
        if abs(f) < tol:
            # Le prix est reproduit, mais encore faut-il que sigma soit identifiable.
            if vega_de(sigma) < VEGA_MIN:
                raise NoImpliedVolatility(
                    f"Vega quasi nul en sigma = {sigma:.4f} : prix insensible à la volatilité "
                    "(option très dans la monnaie)."
                )
            return float(sigma)

        # On resserre l'encadrement à chaque évaluation : il reste toujours valide.
        if f > 0:
            haut = sigma
        else:
            bas = sigma

        vega = vega_de(sigma)
        candidat = sigma - f / vega if vega > VEGA_MIN else np.inf
        sigma = candidat if bas < candidat < haut else 0.5 * (bas + haut)

    raise NoImpliedVolatility(f"Pas de convergence après {max_iter} itérations.")


def _point_de_depart(prix_marche, option, market):
    """Brenner-Subrahmanyam : exact à la monnaie, raisonnable ailleurs."""
    S, T = market.spot, option.maturity
    estimation = np.sqrt(2 * np.pi / T) * prix_marche / S
    return float(np.clip(estimation, 0.05, 2.0))