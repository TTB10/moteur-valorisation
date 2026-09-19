import itertools

import numpy as np
import pytest

from pricer.engines.analytic import BlackScholesEngine
from pricer.instruments import EuropeanOption
from pricer.market_data import MarketData

SPOTS = [50, 80, 100, 120, 200]
STRIKES = [80, 100, 120]
MATURITES = [0.1, 0.5, 1.0, 2.0]
TAUX = [0.0, 0.05]
DIVIDENDES = [0.0, 0.03]
VOLS = [0.1, 0.3, 0.6]

GRILLE = list(itertools.product(SPOTS, STRIKES, MATURITES, TAUX, DIVIDENDES, VOLS))


@pytest.mark.parametrize("S, K, T, r, q, sigma", GRILLE)
def test_parite_call_put(S, K, T, r, q, sigma):
    marche = MarketData(spot=S, rate=r, dividend=q, vol=sigma)
    moteur = BlackScholesEngine()

    call = moteur.price(EuropeanOption(K, T, "call"), marche)
    put = moteur.price(EuropeanOption(K, T, "put"), marche)
    forward_actualise = S * np.exp(-q * T) - K * np.exp(-r * T)

    assert call - put == pytest.approx(forward_actualise, abs=1e-10)


def test_valeur_de_reference():
    """Cas classique des manuels : S = K = 100, T = 1, r = 5 %, q = 0, sigma = 20 %."""
    marche = MarketData(spot=100, rate=0.05, dividend=0.0, vol=0.2)
    moteur = BlackScholesEngine()

    assert moteur.price(EuropeanOption(100, 1.0, "call"), marche) == pytest.approx(10.4506, abs=1e-4)
    assert moteur.price(EuropeanOption(100, 1.0, "put"), marche) == pytest.approx(5.5735, abs=1e-4)