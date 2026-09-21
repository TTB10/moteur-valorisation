import numpy as np
import pandas as pd
import pytest

from pricer.engines.analytic import BlackScholesEngine
from pricer.instruments import EuropeanOption
from pricer.market_data import MarketData
from pricer.vol_surface import (build_vol_surface, butterfly_arbitrage_violations,
                                calendar_arbitrage_violations)

MOTEUR = BlackScholesEngine()
S, R, Q = 100.0, 0.03, 0.01
MATURITES = (0.25, 0.5, 1.0)
STRIKES = range(80, 125, 5)


def chaine_synthetique(vols=None, spread_relatif=0.01):
    """Chaîne complète (calls et puts sur tous les strikes), pricée par Black-Scholes.

    vols : dict maturité -> volatilité ; par défaut 20 % partout (surface plate).
    """
    vols = vols or {T: 0.2 for T in MATURITES}
    lignes = []
    for T in MATURITES:
        marche = MarketData(spot=S, rate=R, dividend=Q, vol=vols[T])
        for K in STRIKES:
            for type_option in ("call", "put"):
                prix = MOTEUR.price(EuropeanOption(float(K), T, type_option), marche)
                lignes.append({"type": type_option, "strike": float(K), "maturity": T,
                               "mid": prix, "spread_relatif": spread_relatif})
    return pd.DataFrame(lignes)


def hors_de_la_monnaie(chaine):
    est_otm = ((chaine["type"] == "call") & (chaine["strike"] >= S)) | \
              ((chaine["type"] == "put") & (chaine["strike"] < S))
    return chaine[est_otm].reset_index(drop=True)


def construire(chaine=None):
    chaine = chaine_synthetique() if chaine is None else chaine
    return build_vol_surface(hors_de_la_monnaie(chaine), S, R, propre_pour_parite=chaine)


def test_surface_plate_retrouve_la_volatilite():
    surface, journal = construire()

    assert journal["points retenus"] == len(hors_de_la_monnaie(chaine_synthetique()))
    assert np.allclose(surface.points["iv"], 0.2, atol=1e-6)


def test_forward_retrouve_par_parite():
    """Le forward estimé doit valoir exactement S e^{(r-q)T}."""
    surface, journal = construire()

    assert journal["forwards estimés par repli (sans parité)"] == 0
    for T in MATURITES:
        assert surface.forwards[T] == pytest.approx(S * np.exp((R - Q) * T), abs=1e-8)


def test_sans_chaine_complete_le_repli_est_journalise():
    """Avec les seules OTM, aucun strike n'est coté en call et en put : repli signalé."""
    surface, journal = build_vol_surface(hors_de_la_monnaie(chaine_synthetique()), S, R)

    assert journal["forwards estimés par repli (sans parité)"] == len(MATURITES)
    assert surface.forwards[1.0] == pytest.approx(S * np.exp(R * 1.0))


def test_structure_par_terme_plate():
    surface, _ = construire()
    atm = surface.atm_term_structure()

    assert len(atm) == len(MATURITES)
    assert np.allclose(atm["iv_atm"], 0.2, atol=1e-6)


def test_aucun_arbitrage_sur_des_prix_black_scholes():
    surface, _ = construire()

    assert all(butterfly_arbitrage_violations(surface, T) == 0 for T in MATURITES)
    assert calendar_arbitrage_violations(surface) == 0


def test_arbitrage_papillon_detecte():
    """On gonfle le prix d'un call central : la convexité en strike est rompue."""
    chaine = chaine_synthetique()
    cible = (chaine["type"] == "call") & (chaine["strike"] == 110) & (chaine["maturity"] == 0.5)
    chaine.loc[cible, "mid"] += 1.0

    surface, _ = construire(chaine)
    assert butterfly_arbitrage_violations(surface, 0.5) >= 1


def test_arbitrage_calendaire_detecte():
    """Variance totale décroissante : 40 % à 3 mois (w = 0,04) puis 20 % à 6 mois (w = 0,02)."""
    chaine = chaine_synthetique(vols={0.25: 0.40, 0.5: 0.20, 1.0: 0.20})

    surface, _ = construire(chaine)
    assert calendar_arbitrage_violations(surface) >= 1