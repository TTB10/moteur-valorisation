import numpy as np
import pandas as pd
import pytest

from pricer.backtest import basel_zone, exceptions, kupiec_test, rolling_var_forecasts


@pytest.mark.parametrize("n_exceptions, rejete", [(2, False), (4, False), (0, True), (10, True)])
def test_kupiec_sur_250_jours(n_exceptions, rejete):
    """Zéro exception est aussi rejeté : le modèle surestime alors le risque."""
    assert kupiec_test(n_exceptions, 250, 0.99).rejected is rejete


def test_kupiec_zero_exception_valeur_exacte():
    """Avec x = 0, LR = -2 N ln(1 - p)."""
    assert kupiec_test(0, 250, 0.99).lr == pytest.approx(-2 * 250 * np.log(0.99))


@pytest.mark.parametrize("n_exceptions, zone", [(0, "vert"), (4, "vert"), (5, "orange"),
                                                 (9, "orange"), (10, "rouge")])
def test_zones_de_bale(n_exceptions, zone):
    assert basel_zone(n_exceptions) == zone


def test_modele_normal_bien_calibre_sur_rendements_normaux():
    """Si les rendements sont vraiment gaussiens, la VaR normale doit donner 1 % d'exceptions."""
    r = pd.Series(0.01 * np.random.default_rng(0).standard_normal(20_000))
    prev = rolling_var_forecasts(r)
    taux = exceptions(prev, "normale").mean()

    assert 0.007 < taux < 0.013


def test_modele_normal_sous_estime_les_queues_epaisses():
    """Sur des rendements de Student à 3 degrés, la VaR normale subit trop d'exceptions."""
    brut = np.random.default_rng(1).standard_t(df=3, size=20_000)
    r = pd.Series(0.01 * brut / np.sqrt(3))            # variance normalisée à 1e-4
    prev = rolling_var_forecasts(r)

    taux_normale = exceptions(prev, "normale").mean()
    taux_historique = exceptions(prev, "historique").mean()

    assert taux_normale > 0.012                        # trop d'exceptions
    assert kupiec_test(int(exceptions(prev, "normale").sum()), len(prev)).rejected
    assert abs(taux_historique - 0.01) < abs(taux_normale - 0.01)


def test_pas_de_biais_d_anticipation():
    """La VaR du premier jour prévu n'utilise que les 250 rendements précédents."""
    r = pd.Series(np.arange(300, dtype=float) / 1000)
    prev = rolling_var_forecasts(r)

    assert prev.index[0] == 250
    attendu = (-r.iloc[:250]).quantile(0.99)
    assert prev["historique"].iloc[0] == pytest.approx(attendu)