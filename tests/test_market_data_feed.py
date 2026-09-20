import numpy as np
import pandas as pd
import pytest

from pricer.market_data_feed import clean_option_chain, implied_forward_by_parity

SPOT = 100.0


def chaine_fabriquee():
    """Une chaîne de test contenant une cotation valide et un défaut de chaque type."""
    return pd.DataFrame([
        # type, strike, maturity, bid, ask, volume, open_interest
        ("call", 105, 0.25, 2.00, 2.10, 500, 1000),   # valide (OTM)
        ("call", 110, 0.25, 1.00, 1.05, 300, 800),    # valide (OTM)
        ("put",   95, 0.25, 1.80, 1.90, 400, 900),    # valide (OTM)
        ("call", 105, 0.25, 0.00, 2.10, 500, 1000),   # bid nul
        ("call", 106, 0.25, 2.10, 2.00, 500, 1000),   # ask < bid
        ("call", 107, 0.25, 2.00, 2.10, 2, 50),       # volume insuffisant
        ("call", 108, 0.25, 1.00, 2.00, 500, 1000),   # spread relatif 67 %
        ("call", 400, 0.25, 0.50, 0.52, 500, 1000),   # strike trop éloigné, mais bien coté
        ("call",  90, 0.25, 11.0, 11.2, 500, 1000),   # dans la monnaie
        ("call", 112, 0.01, 2.00, 2.10, 500, 1000),   # maturité trop courte (3,65 jours)
        ("call", 113, 0.25,  0.04, 0.05, 500, 1000),   # prix trop faible
    ], columns=["type", "strike", "maturity", "bid", "ask", "volume", "open_interest"])


def test_chaque_filtre_rejette_sa_ligne():
    propre, journal = clean_option_chain(chaine_fabriquee(), SPOT)

    assert journal["maturité trop courte"] == 1
    assert journal["cotation absente ou incohérente"] == 2
    assert journal["prix trop faible"] == 1
    assert journal["illiquide (ni volume ni encours)"] == 1
    assert journal["écart achat-vente excessif"] == 1
    assert journal["strike trop éloigné"] == 1
    assert journal["option dans la monnaie (vega faible)"] == 1
    assert journal["total retenu"] == 3


def test_le_mid_est_bien_calcule():
    propre, _ = clean_option_chain(chaine_fabriquee(), SPOT)
    ligne = propre[propre["strike"] == 105].iloc[0]

    assert ligne["mid"] == pytest.approx(2.05)
    assert ligne["spread_relatif"] == pytest.approx(0.10 / 2.05, rel=1e-9)


def test_le_journal_est_coherent():
    propre, journal = clean_option_chain(chaine_fabriquee(), SPOT)
    rejets = sum(v for k, v in journal.items() if k not in ("total initial", "total retenu"))

    assert journal["total initial"] - rejets == journal["total retenu"] == len(propre)


def test_forward_implicite_par_parite():
    """On fabrique des prix cohérents avec F = 102, et on doit le retrouver."""
    r, T, F = 0.04, 0.5, 102.0
    lignes = []
    for K in (95, 100, 105):
        # C - P = e^{-rT}(F - K), on répartit arbitrairement autour
        ecart = np.exp(-r * T) * (F - K)
        base = 5.0
        lignes += [("call", K, T, base + ecart - 0.05, base + ecart + 0.05, 500, 100),
                   ("put", K, T, base - 0.05, base + 0.05, 500, 100)]

    brut = pd.DataFrame(lignes, columns=["type", "strike", "maturity",
                                         "bid", "ask", "volume", "open_interest"])
    propre, _ = clean_option_chain(brut, spot=100.0, garder_otm_seulement=False)

    assert implied_forward_by_parity(propre, T, r) == pytest.approx(F, abs=1e-8)


def test_forward_absent_si_aucun_strike_commun():
    brut = pd.DataFrame([("call", 105, 0.25, 2.0, 2.1, 500, 100)],
                        columns=["type", "strike", "maturity",
                                 "bid", "ask", "volume", "open_interest"])
    propre, _ = clean_option_chain(brut, SPOT)

    assert implied_forward_by_parity(propre, 0.25, 0.04) is None