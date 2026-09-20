import numpy as np
import pytest

from pricer.hedging import simulate_delta_hedge
from pricer.instruments import AmericanOption, EuropeanOption
from pricer.market_data import MarketData

MARCHE = MarketData(spot=100, rate=0.05, dividend=0.0, vol=0.2)
CALL = EuropeanOption(100, 1.0, "call")


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_couverture_sans_biais(option_type):
    """La couverture est autofinancée : l'erreur moyenne doit être nulle."""
    option = EuropeanOption(100, 1.0, option_type)
    res = simulate_delta_hedge(option, MARCHE, n_rebalancements=100, n_paths=20_000, seed=1)

    assert abs(res.mean) < 3 * res.std_error


def test_erreur_decroit_en_racine_de_n():
    """Quadrupler la fréquence doit diviser l'écart-type par environ 2."""
    ecarts = [simulate_delta_hedge(CALL, MARCHE, n, n_paths=20_000, seed=7).std
              for n in (63, 252)]

    assert ecarts[1] / ecarts[0] == pytest.approx(0.5, rel=0.15)


def test_couverture_a_un_pas_est_tres_imprecise():
    """Sans rebalancement, la réplication est un pari sur la trajectoire."""
    res = simulate_delta_hedge(CALL, MARCHE, n_rebalancements=1, n_paths=20_000, seed=3)

    assert res.std > 1.0            # plus de 10 % de la prime
    assert res.quantile(0.01) < -5  # la queue de perte est lourde


def test_vendre_avec_une_vol_trop_faible_fait_perdre():
    """Couvrir à 15 % une action qui bouge à 25 % : le vendeur perd en moyenne."""
    marche_reel = MarketData(spot=100, rate=0.05, dividend=0.0, vol=0.25)
    res = simulate_delta_hedge(CALL, marche_reel, n_rebalancements=252,
                               n_paths=20_000, seed=11, hedge_vol=0.15)

    assert res.mean < -3 * res.std_error


def test_la_derive_reelle_ne_change_pas_la_dispersion():
    """La dérive ne doit pas dégrader la qualité de la couverture."""
    base = simulate_delta_hedge(CALL, MARCHE, 252, n_paths=20_000, seed=5)
    haussier = simulate_delta_hedge(CALL, MARCHE, 252, n_paths=20_000, seed=5, drift=0.30)

    assert haussier.std == pytest.approx(base.std, rel=0.3)
    assert abs(haussier.mean) < 0.10 * haussier.std   # biais négligeable devant la dispersion


def test_biais_de_discretisation_decroit_en_1_sur_N():
    """Sous forte dérive, il reste un biais O(dt) : quadrupler N le divise par 4."""
    biais = [abs(simulate_delta_hedge(CALL, MARCHE, n, n_paths=20_000, seed=5, drift=0.30).mean)
             for n in (63, 252)]

    assert biais[0] / biais[1] == pytest.approx(4.0, rel=0.3)


def test_refuse_une_americaine():
    with pytest.raises(TypeError):
        simulate_delta_hedge(AmericanOption(100, 1.0, "put"), MARCHE, 50, n_paths=100)