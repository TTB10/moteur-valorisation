"""Backtesting de la VaR : prévisions glissantes, exceptions, test de Kupiec, zones de Bâle."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2, norm

SEUIL_KUPIEC = float(chi2.ppf(0.95, df=1))     # 3,841

@dataclass(frozen=True)
class KupiecResult:
    n_obs: int
    n_exceptions: int
    expected: float
    rate: float
    lr: float
    p_value: float

    @property
    def rejected(self):
        """Rejet au seuil de 5 %, dans les deux sens : trop ou trop peu d'exceptions."""
        return bool(self.lr > SEUIL_KUPIEC)


def rolling_var_forecasts(log_returns, confidence=0.99, window=250, lambda_ewma=0.94):
    """Prévisions de VaR (en rendement) calculées avec l'information de la veille.

    Le décalage shift(1) est essentiel : la VaR du jour t ne doit utiliser que les
    rendements jusqu'à t-1, sinon le backtest est contaminé par l'information future.
    """
    r = pd.Series(log_returns, dtype=float)
    pertes = -r
    z = norm.ppf(confidence)

    historique = pertes.rolling(window).quantile(confidence).shift(1)
    normale = z * r.rolling(window).std().shift(1)
    variance_ewma = (r**2).ewm(alpha=1 - lambda_ewma, adjust=False).mean().shift(1)
    ewma = z * np.sqrt(variance_ewma)

    return pd.DataFrame({"perte": pertes, "historique": historique,
                         "normale": normale, "ewma": ewma}).dropna()


def exceptions(previsions, methode):
    """Jours où la perte réalisée dépasse la VaR prévue."""
    return previsions["perte"] > previsions[methode]


def kupiec_test(n_exceptions, n_obs, confidence=0.99):
    """Test du ratio de vraisemblance de Kupiec (proportion of failures)."""
    p = 1 - confidence
    x, n = n_exceptions, n_obs
    p_chapeau = x / n

    # xlogy(a, b) = a ln(b), avec la convention 0 ln 0 = 0 (cas x = 0 ou x = n)
    log_v0 = xlogy(n - x, 1 - p) + xlogy(x, p)
    log_v1 = xlogy(n - x, 1 - p_chapeau) + xlogy(x, p_chapeau)
    lr = float(-2 * (log_v0 - log_v1))

    return KupiecResult(n, x, n * p, p_chapeau, lr, float(chi2.sf(lr, df=1)))


def basel_zone(n_exceptions):
    """Feux de Bâle sur 250 jours à 99 % : vert 0-4, orange 5-9, rouge 10 et plus."""
    if n_exceptions <= 4:
        return "vert"
    if n_exceptions <= 9:
        return "orange"
    return "rouge"