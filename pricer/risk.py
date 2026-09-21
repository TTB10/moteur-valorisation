"""Mesures de risque de marché : VaR et Expected Shortfall par trois méthodes."""

from dataclasses import dataclass, replace

import numpy as np
from scipy.stats import norm

from pricer.instruments import VanillaOption
from pricer.portfolio import Portfolio, Position

JOURS_PAR_AN = 252


@dataclass(frozen=True)
class RiskResult:
    """VaR et ES d'un portefeuille, avec les scénarios de perte quand ils existent."""

    method: str
    confidence: float
    var: float
    es: float
    losses: np.ndarray = None      # absent pour la méthode paramétrique


def var_es(losses, confidence):
    """VaR empirique (quantile) et ES (moyenne des k pires pertes, k = (1 - alpha) n).

    L'ES est calculée sur les k pires scénarios plutôt que sur les pertes >= VaR : avec des
    pertes discrètes (ex-aequo au niveau de la VaR), la seconde définition est biaisée.
    """
    losses = np.asarray(losses, dtype=float)
    var = float(np.quantile(losses, confidence))
    k = max(1, int(np.ceil((1 - confidence) * len(losses))))
    es = float(np.sort(losses)[-k:].mean())
    return var, es


def age(instrument, horizon):
    """Vieillit un instrument : les options perdent 'horizon' années de maturité."""
    if isinstance(instrument, Portfolio):
        positions = tuple(Position(age(p.instrument, horizon), p.quantity)
                          for p in instrument.positions)
        return replace(instrument, positions=positions)
    if isinstance(instrument, VanillaOption):
        return replace(instrument, maturity=max(instrument.maturity - horizon, 1e-6))
    return instrument


def scenario_losses(portfolio, market, log_returns, horizon_days=1):
    """Pertes par réévaluation complète : spot choqué, portefeuille vieilli de l'horizon."""
    v0 = portfolio.price(market)
    vieilli = age(portfolio, horizon_days / JOURS_PAR_AN)
    spots = market.spot * np.exp(np.asarray(log_returns, dtype=float))
    valeurs = np.array([vieilli.price(replace(market, spot=s)) for s in spots])
    return v0 - valeurs


def historical_var(portfolio, market, log_returns, confidence=0.99):
    """On rejoue les rendements historiques sur le portefeuille actuel."""
    pertes = scenario_losses(portfolio, market, log_returns)
    var, es = var_es(pertes, confidence)
    return RiskResult("historique", confidence, var, es, pertes)


def parametric_var(portfolio, market, daily_vol, confidence=0.99):
    """Delta-normale : L ≈ -Δ S r avec r ~ N(0, σ²). Formule fermée, mais ignore le gamma."""
    delta = portfolio.greeks(market).delta
    ecart_type = abs(delta) * market.spot * daily_vol
    z = norm.ppf(confidence)
    return RiskResult("paramétrique delta-normale", confidence,
                      z * ecart_type, ecart_type * norm.pdf(z) / (1 - confidence))


def monte_carlo_var(portfolio, market, daily_vol, confidence=0.99,
                    n_scenarios=20_000, seed=None):
    """Rendements lognormaux simulés, réévaluation complète du portefeuille."""
    rng = np.random.default_rng(seed)
    rendements = -0.5 * daily_vol**2 + daily_vol * rng.standard_normal(n_scenarios)
    pertes = scenario_losses(portfolio, market, rendements)
    var, es = var_es(pertes, confidence)
    return RiskResult("Monte Carlo (réévaluation complète)", confidence, var, es, pertes)