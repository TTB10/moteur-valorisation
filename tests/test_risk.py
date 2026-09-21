import numpy as np
import pytest
from scipy.stats import norm

from pricer.instruments import EuropeanOption, Underlying
from pricer.market_data import MarketData
from pricer.portfolio import Portfolio
from pricer.risk import (age, historical_var, monte_carlo_var, parametric_var, var_es)
from pricer.strategies import straddle

MARCHE = MarketData(spot=100, rate=0.03, dividend=0.0, vol=0.2)
VOL_JOUR = 0.2 / np.sqrt(252)


def test_var_et_es_d_une_loi_normale():
    """VaR 99 % = 2,326 sigma et ES 99 % = 2,665 sigma pour une perte gaussienne."""
    pertes = np.random.default_rng(0).standard_normal(400_000)
    var, es = var_es(pertes, 0.99)
    z = norm.ppf(0.99)

    assert var == pytest.approx(z, rel=0.02)
    assert es == pytest.approx(norm.pdf(z) / 0.01, rel=0.02)


def test_es_toujours_superieure_a_la_var():
    pertes = np.random.default_rng(1).standard_t(df=3, size=100_000)
    for alpha in (0.95, 0.975, 0.99):
        var, es = var_es(pertes, alpha)
        assert es >= var


def test_position_lineaire_les_trois_methodes_concordent():
    """Sans convexité, la méthode delta-normale est exacte à l'ordre 1."""
    pf = Portfolio.from_pairs([(Underlying(), 10)])
    rendements = VOL_JOUR * np.random.default_rng(2).standard_normal(100_000)

    param = parametric_var(pf, MARCHE, VOL_JOUR)
    mc = monte_carlo_var(pf, MARCHE, VOL_JOUR, n_scenarios=100_000, seed=3)
    hist = historical_var(pf, MARCHE, rendements)

    assert mc.var == pytest.approx(param.var, rel=0.03)
    assert hist.var == pytest.approx(param.var, rel=0.03)


def test_delta_normale_aveugle_a_la_convexite():
    """Straddle vendu couvert en delta : VaR paramétrique nulle, alors que le risque est réel."""
    pf = Portfolio.from_pairs([(straddle(100, 0.25), -1)]).delta_hedged(MARCHE)

    param = parametric_var(pf, MARCHE, VOL_JOUR)
    mc = monte_carlo_var(pf, MARCHE, VOL_JOUR, n_scenarios=10_000, seed=4)

    assert param.var == pytest.approx(0.0, abs=1e-9)
    assert mc.var > 0.1


def test_var_non_sous_additive_mais_es_sous_additive():
    """Deux pertes indépendantes de 100 avec probabilité 4 %, au seuil de 95 %."""
    rng = np.random.default_rng(5)
    a = 100.0 * (rng.random(1_000_000) < 0.04)
    b = 100.0 * (rng.random(1_000_000) < 0.04)

    var_a, es_a = var_es(a, 0.95)
    var_b, es_b = var_es(b, 0.95)
    var_ab, es_ab = var_es(a + b, 0.95)

    assert var_a == 0 and var_b == 0
    assert var_ab == 100                  # diversifier a augmenté la VaR
    assert es_ab <= es_a + es_b           # l'ES, elle, respecte la diversification


def test_vieillissement_reduit_toutes_les_maturites():
    vieilli = age(straddle(100, 1.0), 0.1)
    maturites = [p.instrument.maturity for p in vieilli.positions]

    assert maturites == pytest.approx([0.9, 0.9])
    assert age(EuropeanOption(100, 1.0), 0.25).maturity == pytest.approx(0.75)