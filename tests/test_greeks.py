import itertools

import numpy as np
import pytest

from pricer.engines.analytic import BlackScholesEngine
from pricer.instruments import EuropeanOption
from pricer.market_data import MarketData

MOTEUR = BlackScholesEngine()

GRILLE = list(itertools.product([80, 100, 120], [100], [0.25, 1.0, 2.0],
                                [0.0, 0.05], [0.0, 0.03], [0.15, 0.4]))


def marche_et_options(S, K, T, r, q, sigma):
    marche = MarketData(spot=S, rate=r, dividend=q, vol=sigma)
    return marche, EuropeanOption(K, T, "call"), EuropeanOption(K, T, "put")


@pytest.mark.parametrize("S, K, T, r, q, sigma", GRILLE)
def test_relations_de_parite_sur_les_grecques(S, K, T, r, q, sigma):
    """En dérivant la parité C - P = S e^-qT - K e^-rT."""
    marche, call, put = marche_et_options(S, K, T, r, q, sigma)
    gc = MOTEUR.greeks(call, marche)
    gp = MOTEUR.greeks(put, marche)

    assert gc.delta - gp.delta == pytest.approx(np.exp(-q * T), abs=1e-12)
    assert gc.gamma == pytest.approx(gp.gamma, abs=1e-12)
    assert gc.vega == pytest.approx(gp.vega, abs=1e-12)
    assert gc.rho - gp.rho == pytest.approx(K * T * np.exp(-r * T), abs=1e-10)


@pytest.mark.parametrize("S, K, T, r, q, sigma", GRILLE)
def test_signes_et_bornes(S, K, T, r, q, sigma):
    marche, call, put = marche_et_options(S, K, T, r, q, sigma)
    gc = MOTEUR.greeks(call, marche)
    gp = MOTEUR.greeks(put, marche)

    assert 0 < gc.delta < np.exp(-q * T)      # borné par l'escompte du dividende
    assert -np.exp(-q * T) < gp.delta < 0
    assert gc.gamma > 0 and gc.vega > 0       # toute option longue est convexe
    assert gc.rho > 0 and gp.rho < 0


def test_valeurs_de_reference():
    """S = K = 100, T = 1, r = 5 %, q = 0, sigma = 20 %."""
    marche = MarketData(spot=100, rate=0.05, dividend=0.0, vol=0.2)
    g = MOTEUR.greeks(EuropeanOption(100, 1.0, "call"), marche)

    assert g.delta == pytest.approx(0.6368, abs=1e-4)
    assert g.gamma == pytest.approx(0.018762, abs=1e-6)
    assert g.vega == pytest.approx(37.524, abs=1e-3)
    assert g.theta == pytest.approx(-6.414, abs=1e-3)
    assert g.rho == pytest.approx(53.2325, abs=1e-4)


def test_gamma_en_cloche_autour_de_la_monnaie():
    """Le gamma est en cloche autour de la monnaie. Son maximum exact est en d1 = -sigma*sqrt(T),
    soit légèrement sous le strike quand r > 0 : ce test ne vérifie que la forme en cloche."""
    marche_ref = MarketData(spot=100, rate=0.0, dividend=0.0, vol=0.2)
    option = EuropeanOption(100, 1.0)

    gammas = {
        S: MOTEUR.greeks(option, MarketData(spot=S, rate=0.0, dividend=0.0, vol=0.2)).gamma
        for S in (60, 80, 100, 130, 160)
    }
    assert gammas[100] == max(gammas.values())
    assert gammas[60] < gammas[80] < gammas[100]
    assert gammas[160] < gammas[130] < gammas[100]


def test_grecques_refusent_une_americaine():
    from pricer.instruments import AmericanOption

    marche = MarketData(spot=100, rate=0.05, dividend=0.0, vol=0.2)
    with pytest.raises(TypeError):
        MOTEUR.greeks(AmericanOption(100, 1.0, "put"), marche)