import itertools

import pytest

from pricer.engines.analytic import BlackScholesEngine
from pricer.engines.binomial import BinomialTreeEngine
from pricer.engines.monte_carlo import MonteCarloEngine
from pricer.finite_difference import finite_difference_greeks
from pricer.instruments import AmericanOption, EuropeanOption
from pricer.market_data import MarketData

MOTEUR_BS = BlackScholesEngine()

GRILLE = list(itertools.product([90, 100, 110], [100], [0.5, 1.0], [0.03], [0.0, 0.02], [0.2]))


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("S, K, T, r, q, sigma", GRILLE)
def test_fd_coincide_avec_analytique(S, K, T, r, q, sigma, option_type):
    marche = MarketData(spot=S, rate=r, dividend=q, vol=sigma)
    option = EuropeanOption(K, T, option_type)

    exact = MOTEUR_BS.greeks(option, marche)
    approx = finite_difference_greeks(MOTEUR_BS, option, marche)

    assert approx.delta == pytest.approx(exact.delta, abs=1e-6)
    assert approx.gamma == pytest.approx(exact.gamma, rel=1e-4)
    assert approx.vega == pytest.approx(exact.vega, rel=1e-6)
    assert approx.theta == pytest.approx(exact.theta, rel=1e-6)
    assert approx.rho == pytest.approx(exact.rho, rel=1e-6)


def test_fd_sur_arbre_retrouve_black_scholes():
    """L'arbre n'a pas de formule de grecques : on les obtient par différences finies."""
    marche = MarketData(spot=100, rate=0.05, dividend=0.0, vol=0.2)
    option = EuropeanOption(100, 1.0)

    exact = MOTEUR_BS.greeks(option, marche)
    approx = finite_difference_greeks(BinomialTreeEngine(n_steps=2000), option, marche)

    assert approx.delta == pytest.approx(exact.delta, abs=1e-3)
    assert approx.vega == pytest.approx(exact.vega, rel=1e-2)


def test_fd_sur_monte_carlo_avec_nombres_aleatoires_communs():
    """La graine fixée garantit les mêmes tirages pour les deux évaluations."""
    marche = MarketData(spot=100, rate=0.05, dividend=0.0, vol=0.2)
    option = EuropeanOption(100, 1.0)

    exact = MOTEUR_BS.greeks(option, marche)
    approx = finite_difference_greeks(MonteCarloEngine(n_paths=400_000, seed=11), option, marche)

    assert approx.delta == pytest.approx(exact.delta, abs=5e-3)


def test_delta_du_put_americain_domine_celui_de_l_europeen():
    """L'exercice anticipé rend le put plus sensible au spot."""
    marche = MarketData(spot=100, rate=0.05, dividend=0.0, vol=0.2)
    moteur = BinomialTreeEngine(n_steps=2000)

    europeen = finite_difference_greeks(moteur, EuropeanOption(100, 1.0, "put"), marche)
    americain = finite_difference_greeks(moteur, AmericanOption(100, 1.0, "put"), marche)

    assert americain.delta < europeen.delta


def test_pas_de_temps_trop_grand_refuse():
    marche = MarketData(spot=100, rate=0.05, dividend=0.0, vol=0.2)
    with pytest.raises(ValueError):
        finite_difference_greeks(MOTEUR_BS, EuropeanOption(100, 0.05), marche, pas_temps=0.1)


def test_erreur_du_gamma_decroit_en_h_carre():
    """Diviser le pas par 2 doit diviser l'erreur de troncature par 4."""
    marche = MarketData(spot=100, rate=0.03, dividend=0.0, vol=0.2)
    option = EuropeanOption(100, 1.0)
    exact = MOTEUR_BS.greeks(option, marche).gamma

    erreurs = [
        abs(finite_difference_greeks(MOTEUR_BS, option, marche, pas_spot_gamma=p).gamma - exact)
        for p in (2e-2, 1e-2)
    ]
    assert erreurs[0] / erreurs[1] == pytest.approx(4.0, rel=0.1)