import pytest

from pricer.engines.analytic import BlackScholesEngine
from pricer.engines.monte_carlo import MonteCarloEngine
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
def test_monte_carlo_retrouve_black_scholes(S, K, T, r, q, sigma, option_type):
    marche = MarketData(spot=S, rate=r, dividend=q, vol=sigma)
    option = EuropeanOption(K, T, option_type)

    prix_bs = BlackScholesEngine().price(option, marche)
    resultat = MonteCarloEngine(n_paths=200_000, seed=2026).simulate(option, marche)

    assert abs(resultat.price - prix_bs) < 3 * resultat.std_error


def test_erreur_standard_en_racine_de_n():
    """4 fois plus de tirages doivent diviser l'erreur standard par 2."""
    marche = MarketData(spot=100, rate=0.05, dividend=0.0, vol=0.2)
    option = EuropeanOption(100, 1.0)

    erreur_1 = MonteCarloEngine(n_paths=100_000, seed=1).simulate(option, marche).std_error
    erreur_4 = MonteCarloEngine(n_paths=400_000, seed=1).simulate(option, marche).std_error

    assert erreur_4 / erreur_1 == pytest.approx(0.5, rel=0.05)

@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("S, K, T, r, q, sigma", CAS)
def test_antithetique_retrouve_black_scholes(S, K, T, r, q, sigma, option_type):
    marche = MarketData(spot=S, rate=r, dividend=q, vol=sigma)
    option = EuropeanOption(K, T, option_type)

    prix_bs = BlackScholesEngine().price(option, marche)
    resultat = MonteCarloEngine(n_paths=200_000, seed=2026, antithetic=True).simulate(option, marche)

    assert abs(resultat.price - prix_bs) < 3 * resultat.std_error


def test_antithetique_reduit_la_variance():
    """À nombre égal de spots simulés, l'erreur standard doit baisser nettement."""
    marche = MarketData(spot=100, rate=0.05, dividend=0.0, vol=0.2)
    option = EuropeanOption(100, 1.0)

    simple = MonteCarloEngine(n_paths=200_000, seed=7).simulate(option, marche)
    anti = MonteCarloEngine(n_paths=200_000, seed=7, antithetic=True).simulate(option, marche)

    assert (simple.std_error / anti.std_error) ** 2 > 1.5

@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("S, K, T, r, q, sigma", CAS)
def test_controle_retrouve_black_scholes(S, K, T, r, q, sigma, option_type):
    marche = MarketData(spot=S, rate=r, dividend=q, vol=sigma)
    option = EuropeanOption(K, T, option_type)

    prix_bs = BlackScholesEngine().price(option, marche)
    resultat = MonteCarloEngine(n_paths=200_000, seed=2026, control_variate=True).simulate(option, marche)

    assert abs(resultat.price - prix_bs) < 3 * resultat.std_error


def test_controle_reduit_la_variance():
    """À nombre égal de spots simulés, le gain doit être nettement supérieur à 1."""
    marche = MarketData(spot=100, rate=0.05, dividend=0.0, vol=0.2)
    option = EuropeanOption(100, 1.0)

    simple = MonteCarloEngine(n_paths=200_000, seed=7).simulate(option, marche)
    controle = MonteCarloEngine(n_paths=200_000, seed=7, control_variate=True).simulate(option, marche)

    assert (simple.std_error / controle.std_error) ** 2 > 2