import numpy as np
import pytest

from pricer.instruments import AmericanOption, EuropeanOption
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

@pytest.mark.parametrize("S, K, T, r, q, sigma", [c for c in CAS if c[4] == 0])
def test_call_americain_sans_dividende_egal_europeen(S, K, T, r, q, sigma):
    """Sans dividende, exercer un call en avance n'est jamais optimal."""
    marche = MarketData(spot=S, rate=r, dividend=q, vol=sigma)
    moteur = BinomialTreeEngine(n_steps=500)

    europeen = moteur.price(EuropeanOption(K, T, "call"), marche)
    americain = moteur.price(AmericanOption(K, T, "call"), marche)

    assert americain == pytest.approx(europeen, abs=1e-10)


@pytest.mark.parametrize("S, K, T, r, q, sigma", CAS)
def test_put_americain_domine(S, K, T, r, q, sigma):
    """Le put américain vaut au moins l'européen et au moins sa valeur d'exercice immédiat."""
    marche = MarketData(spot=S, rate=r, dividend=q, vol=sigma)
    moteur = BinomialTreeEngine(n_steps=500)

    europeen = moteur.price(EuropeanOption(K, T, "put"), marche)
    americain = moteur.price(AmericanOption(K, T, "put"), marche)

    assert americain >= europeen - 1e-12
    assert americain >= max(K - S, 0.0) - 1e-12


def test_put_tres_dans_la_monnaie_exerce_immediatement():
    """Spot très bas et taux élevé : mieux vaut encaisser K tout de suite."""
    marche = MarketData(spot=50, rate=0.10, dividend=0.0, vol=0.2)
    prix = BinomialTreeEngine(n_steps=500).price(AmericanOption(100, 1.0, "put"), marche)

    assert prix == pytest.approx(50.0, abs=1e-10)


def test_black_scholes_refuse_americaine():
    marche = MarketData(spot=100, rate=0.05, dividend=0.0, vol=0.2)
    with pytest.raises(TypeError):
        BlackScholesEngine().price(AmericanOption(100, 1.0, "put"), marche)