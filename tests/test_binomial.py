import numpy as np
import pytest

from pricer.engines.analytic import BlackScholesEngine
from pricer.engines.binomial import BinomialTreeEngine
from pricer.instruments import EuropeanOption
from pricer.market_data import MarketData

CAS = [
    (100, 100, 1.0, 0.05, 0.00, 0.20),
    (100, 120, 0.5, 0.03, 0.02, 0.30),
    (80, 100, 2.0, 0.01, 0.00, 0.50),
    (120, 100, 0.25, 0.05, 0.03, 0.15),
]


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("S, K, T, r, q, sigma", CAS)
def test_arbre_converge_vers_black_scholes(S, K, T, r, q, sigma, option_type):
    marche = MarketData(spot=S, rate=r, dividend=q, vol=sigma)
    option = EuropeanOption(K, T, option_type)

    prix_bs = BlackScholesEngine().price(option, marche)
    prix_arbre = BinomialTreeEngine(n_steps=2000).price(option, marche)

    assert prix_arbre == pytest.approx(prix_bs, abs=0.01)


@pytest.mark.parametrize("S, K, T, r, q, sigma", CAS)
def test_parite_dans_l_arbre(S, K, T, r, q, sigma):
    """p rend l'action actualisée neutre en moyenne : la parité tient exactement dans l'arbre."""
    marche = MarketData(spot=S, rate=r, dividend=q, vol=sigma)
    moteur = BinomialTreeEngine(n_steps=500)

    call = moteur.price(EuropeanOption(K, T, "call"), marche)
    put = moteur.price(EuropeanOption(K, T, "put"), marche)

    assert call - put == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-8)


def test_arbitrage_detecte():
    """Taux élevé, volatilité faible et un seul pas : p sort de [0, 1], le moteur doit refuser."""
    marche = MarketData(spot=100, rate=0.5, dividend=0.0, vol=0.01)
    with pytest.raises(ValueError):
        BinomialTreeEngine(n_steps=1).price(EuropeanOption(100, 1.0), marche)