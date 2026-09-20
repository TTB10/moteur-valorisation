import numpy as np
import pytest

from pricer.engines.analytic import BlackScholesEngine
from pricer.instruments import EuropeanOption, Underlying
from pricer.market_data import MarketData
from pricer.portfolio import Portfolio
from pricer.strategies import bear_spread, bull_spread, butterfly, straddle, strangle

MARCHE = MarketData(spot=100, rate=0.05, dividend=0.0, vol=0.2)
MOTEUR = BlackScholesEngine()


def test_portefeuille_additif_sur_le_prix():
    call = EuropeanOption(100, 1.0, "call")
    put = EuropeanOption(100, 1.0, "put")
    pf = Portfolio.from_pairs([(call, 3), (put, -2)])

    attendu = 3 * MOTEUR.price(call, MARCHE) - 2 * MOTEUR.price(put, MARCHE)
    assert pf.price(MARCHE) == pytest.approx(attendu, abs=1e-12)


def test_portefeuille_additif_sur_les_grecques():
    call = EuropeanOption(110, 0.5, "call")
    put = EuropeanOption(90, 0.5, "put")
    pf = Portfolio.from_pairs([(call, 2), (put, 1.5)])

    gc, gp, g = MOTEUR.greeks(call, MARCHE), MOTEUR.greeks(put, MARCHE), pf.greeks(MARCHE)
    assert g.delta == pytest.approx(2 * gc.delta + 1.5 * gp.delta, abs=1e-12)
    assert g.vega == pytest.approx(2 * gc.vega + 1.5 * gp.vega, abs=1e-12)


def test_call_moins_put_est_un_forward():
    """Parité vue comme un portefeuille : C - P = S - K e^-rT."""
    pf = Portfolio.from_pairs([(EuropeanOption(100, 1.0, "call"), 1),
                               (EuropeanOption(100, 1.0, "put"), -1)])

    attendu = MARCHE.spot - 100 * np.exp(-MARCHE.rate * 1.0)
    assert pf.price(MARCHE) == pytest.approx(attendu, abs=1e-10)
    assert pf.greeks(MARCHE).delta == pytest.approx(1.0, abs=1e-12)
    assert pf.greeks(MARCHE).gamma == pytest.approx(0.0, abs=1e-14)


def test_couverture_en_delta_annule_le_delta():
    pf = straddle(100, 1.0).delta_hedged(MARCHE)
    assert pf.greeks(MARCHE).delta == pytest.approx(0.0, abs=1e-12)
    assert pf.greeks(MARCHE).gamma > 0          # le gamma, lui, reste


def test_payoff_du_bull_spread_est_borne():
    spots = np.linspace(50, 150, 201)
    payoff = bull_spread(95, 105, 1.0).payoff(spots)

    assert payoff.min() == pytest.approx(0.0)
    assert payoff.max() == pytest.approx(10.0)  # écart des strikes


def test_straddle_double_le_gamma_et_le_vega():
    call = EuropeanOption(100, 1.0, "call")
    g_option, g_straddle = MOTEUR.greeks(call, MARCHE), straddle(100, 1.0).greeks(MARCHE)

    assert g_straddle.gamma == pytest.approx(2 * g_option.gamma, abs=1e-12)
    assert g_straddle.vega == pytest.approx(2 * g_option.vega, abs=1e-12)


def test_butterfly_est_short_gamma_au_centre_et_peu_cher():
    pf = butterfly(90, 100, 110, 1.0)

    assert pf.greeks(MARCHE).gamma < 0
    assert 0 < pf.price(MARCHE) < 5
    assert pf.payoff(np.array([100.0]))[0] == pytest.approx(10.0)   # gain maximal au centre


def test_strangle_moins_cher_que_le_straddle():
    assert strangle(90, 110, 1.0).price(MARCHE) < straddle(100, 1.0).price(MARCHE)


def test_spreads_sont_de_signes_opposes_en_delta():
    assert bull_spread(95, 105, 1.0).greeks(MARCHE).delta > 0
    assert bear_spread(95, 105, 1.0).greeks(MARCHE).delta < 0


def test_portefeuille_imbrique():
    """Un portefeuille peut contenir un portefeuille : patron Composite."""
    interne = straddle(100, 1.0)
    externe = Portfolio.from_pairs([(interne, 2), (Underlying(), -1)])

    attendu = 2 * interne.price(MARCHE) - MARCHE.spot
    assert externe.price(MARCHE) == pytest.approx(attendu, abs=1e-12)
    assert externe.greeks(MARCHE).delta == pytest.approx(2 * interne.greeks(MARCHE).delta - 1,
                                                         abs=1e-12)


def test_strategies_refusent_des_strikes_incoherents():
    with pytest.raises(ValueError):
        bull_spread(105, 95, 1.0)
    with pytest.raises(ValueError):
        butterfly(100, 90, 110, 1.0)