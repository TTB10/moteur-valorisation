"""Simulation de couverture dynamique en delta : réplication discrète d'un call vendu."""

from dataclasses import dataclass, replace

import numpy as np

from pricer.engines.analytic import BlackScholesEngine
from pricer.instruments import EuropeanOption

MOTEUR = BlackScholesEngine()


@dataclass(frozen=True)
class HedgeResult:
    """Résultat d'une expérience de couverture sur un grand nombre de trajectoires."""

    errors: np.ndarray      # erreur finale de chaque trajectoire
    n_rebalancements: int
    premium: float          # prime encaissée à t = 0

    @property
    def mean(self):
        return float(self.errors.mean())

    @property
    def std(self):
        return float(self.errors.std(ddof=1))

    @property
    def std_error(self):
        """Incertitude Monte Carlo sur la moyenne."""
        return self.std / np.sqrt(len(self.errors))

    def quantile(self, q):
        return float(np.quantile(self.errors, q))


def simulate_delta_hedge(option, market, n_rebalancements, n_paths=20_000,
                         seed=None, hedge_vol=None, drift=None):
    """Vend l'option, la couvre en delta à fréquence fixe, renvoie l'erreur finale.

    hedge_vol : volatilité utilisée pour le delta (défaut : celle du marché).
    drift     : dérive réelle mu de la simulation (défaut : r, soit la mesure risque-neutre).
    """
    if not isinstance(option, EuropeanOption):
        raise TypeError("La couverture n'est implémentée que pour les options européennes.")

    S0, r, q = market.spot, market.rate, market.dividend
    sigma_reelle = market.vol                                   # volatilité qui génère les prix
    sigma_couv = market.vol if hedge_vol is None else hedge_vol  # volatilité qui calcule le delta
    mu = r if drift is None else drift
    T = option.maturity
    dt = T / n_rebalancements

    marche_couv = replace(market, vol=sigma_couv)
    rng = np.random.default_rng(seed)

    # 1. Trajectoires du sous-jacent sous la probabilité réelle, en une seule matrice
    Z = rng.standard_normal((n_paths, n_rebalancements))
    increments = (mu - q - 0.5 * sigma_reelle**2) * dt + sigma_reelle * np.sqrt(dt) * Z
    spots = S0 * np.exp(np.cumsum(increments, axis=1))          # spots aux dates 1..N
    spots = np.column_stack([np.full(n_paths, S0), spots])      # on ajoute S0 en tête

    # 2. Position initiale : on encaisse la prime et on achète delta_0 actions
    prime = MOTEUR.price(option, marche_couv)
    delta = np.full(n_paths, MOTEUR.greeks(option, marche_couv).delta)
    cash = prime - delta * S0

    # 3. Rebalancements aux dates 1 .. N-1
    for i in range(1, n_rebalancements):
        temps_restant = T - i * dt
        cash *= np.exp(r * dt)                                  # le compte capitalise
        cash += delta * spots[:, i] * (np.exp(q * dt) - 1)      # dividendes reçus
        nouveau_delta = _delta_vectorise(option, marche_couv, spots[:, i], temps_restant)
        cash -= (nouveau_delta - delta) * spots[:, i]           # achat ou vente, autofinancé
        delta = nouveau_delta

    # 4. Liquidation à maturité
    cash *= np.exp(r * dt)
    cash += delta * spots[:, -1] * (np.exp(q * dt) - 1)
    valeur_finale = cash + delta * spots[:, -1]
    errors = valeur_finale - option.payoff(spots[:, -1])

    return HedgeResult(errors, n_rebalancements, float(prime))


def _delta_vectorise(option, market, spots, temps_restant):
    """Delta de Black-Scholes pour un vecteur de spots, à maturité résiduelle donnée."""
    from scipy.stats import norm

    K, sigma, r, q = option.strike, market.vol, market.rate, market.dividend
    d1, _ = BlackScholesEngine.d1_d2(spots, K, temps_restant, r, q, sigma)
    escompte = np.exp(-q * temps_restant)
    if option.option_type == "call":
        return escompte * norm.cdf(d1)
    return -escompte * norm.cdf(-d1)