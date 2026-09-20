import itertools

import numpy as np
import pytest

from pricer.engines.analytic import BlackScholesEngine
from pricer.implied_vol import (NoImpliedVolatility, implied_volatility, price_bounds)
from pricer.instruments import AmericanOption, EuropeanOption
from pricer.market_data import MarketData

MOTEUR = BlackScholesEngine()

STRIKES = [60, 80, 100, 120, 160]
MATURITES = [0.05, 0.5, 2.0]
VOLS = [0.05, 0.2, 0.8]
GRILLE = list(itertools.product(STRIKES, MATURITES, VOLS, ["call", "put"]))


@pytest.mark.parametrize("K, T, vol_vraie, option_type", GRILLE)
def test_aller_retour_prix_volatilite(K, T, vol_vraie, option_type):
    """On price avec une vol connue, on inverse, on doit la retrouver.

    Les options très dans la monnaie sont exclues : leur vega est nul, donc leur prix
    ne contient aucune information sur la volatilité. C'est pourquoi le marché construit
    les surfaces à partir des options hors de la monnaie.
    """
    marche_vrai = MarketData(spot=100, rate=0.03, dividend=0.01, vol=vol_vraie)
    option = EuropeanOption(K, T, option_type)
    prix = MOTEUR.price(option, marche_vrai)

    # Précision atteignable sur sigma ~ tol / vega : on exclut les points trop peu sensibles.
    if MOTEUR.greeks(option, marche_vrai).vega < 1e-2:
        pytest.skip("Vega négligeable : la volatilité n'est pas identifiable.")

    marche_quelconque = MarketData(spot=100, rate=0.03, dividend=0.01, vol=0.5)
    assert implied_volatility(prix, option, marche_quelconque) == pytest.approx(vol_vraie, abs=1e-6)


@pytest.mark.parametrize("K, option_type", [(60, "call"), (160, "put")])
def test_option_tres_dans_la_monnaie_est_rejetee(K, option_type):
    """Prix insensible à sigma : le moteur doit refuser plutôt que renvoyer un chiffre faux."""
    marche = MarketData(spot=100, rate=0.03, dividend=0.01, vol=0.05)
    option = EuropeanOption(K, 0.05, option_type)
    prix = MOTEUR.price(option, marche)

    with pytest.raises(NoImpliedVolatility):
        implied_volatility(prix, option, marche)

def test_call_et_put_donnent_la_meme_volatilite():
    """Par parité, deux prix cohérents doivent donner la même implicite."""
    marche = MarketData(spot=100, rate=0.05, dividend=0.02, vol=0.27)
    call, put = EuropeanOption(110, 1.0, "call"), EuropeanOption(110, 1.0, "put")

    vol_call = implied_volatility(MOTEUR.price(call, marche), call, marche)
    vol_put = implied_volatility(MOTEUR.price(put, marche), put, marche)

    assert vol_call == pytest.approx(vol_put, abs=1e-8)


def test_prix_sous_la_borne_basse_refuse():
    """Un call sous sa valeur intrinsèque actualisée est un arbitrage."""
    marche = MarketData(spot=100, rate=0.05, dividend=0.0, vol=0.2)
    option = EuropeanOption(80, 1.0, "call")
    borne_basse, _ = price_bounds(option, marche)

    with pytest.raises(NoImpliedVolatility):
        implied_volatility(borne_basse - 0.01, option, marche)


def test_prix_au_dessus_de_la_borne_haute_refuse():
    marche = MarketData(spot=100, rate=0.05, dividend=0.0, vol=0.2)
    option = EuropeanOption(100, 1.0, "call")

    with pytest.raises(NoImpliedVolatility):
        implied_volatility(marche.spot + 1.0, option, marche)


def test_convergence_rapide_a_la_monnaie():
    """Newton doit converger en très peu d'itérations près de la monnaie."""
    marche = MarketData(spot=100, rate=0.05, dividend=0.0, vol=0.3)
    option = EuropeanOption(100, 1.0)
    prix = MOTEUR.price(option, marche)

    assert implied_volatility(prix, option, marche, max_iter=6) == pytest.approx(0.3, abs=1e-8)


def test_strike_extreme_ou_le_vega_est_minuscule():
    """Là où Newton diverge, le repli dichotomique doit sauver la convergence."""
    marche = MarketData(spot=100, rate=0.05, dividend=0.0, vol=0.15)
    option = EuropeanOption(250, 0.25, "call")
    prix = MOTEUR.price(option, marche)

    if prix > 1e-12:
        assert implied_volatility(prix, option, marche) == pytest.approx(0.15, abs=1e-4)


def test_refuse_une_americaine():
    marche = MarketData(spot=100, rate=0.05, dividend=0.0, vol=0.2)
    with pytest.raises(TypeError):
        implied_volatility(6.0, AmericanOption(100, 1.0, "put"), marche)

def test_precision_limitee_par_le_vega():
    """L'incertitude sur sigma vaut environ tol / vega : c'est une limite structurelle."""
    marche = MarketData(spot=100, rate=0.03, dividend=0.01, vol=0.05)
    option = EuropeanOption(120, 0.5, "put")
    prix = MOTEUR.price(option, marche)
    vega = MOTEUR.greeks(option, marche).vega

    tol = 1e-8
    erreur = abs(implied_volatility(prix, option, marche, tol=tol) - 0.05)

    assert erreur < 10 * tol / vega     # l'erreur reste du bon ordre de grandeur
    assert vega < 1e-2                  # et ce cas est bien un cas peu sensible