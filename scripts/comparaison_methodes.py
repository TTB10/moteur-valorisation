"""Tableau comparatif des trois méthodes : prix, écart à Black-Scholes, précision, temps."""

import time

from pricer.engines.analytic import BlackScholesEngine
from pricer.engines.binomial import BinomialTreeEngine
from pricer.engines.monte_carlo import MonteCarloEngine
from pricer.instruments import EuropeanOption
from pricer.market_data import MarketData

marche = MarketData(spot=100, rate=0.05, dividend=0.0, vol=0.2)
option = EuropeanOption(strike=100, maturity=1.0)
prix_bs = BlackScholesEngine().price(option, marche)


def chronometrer(fonction, repetitions=5):
    """Exécute plusieurs fois et garde le meilleur temps, pour limiter le bruit de l'ordinateur."""
    meilleur = float("inf")
    for _ in range(repetitions):
        debut = time.perf_counter()
        resultat = fonction()
        meilleur = min(meilleur, time.perf_counter() - debut)
    return resultat, meilleur * 1000   # en millisecondes


def monte_carlo(**reglages):
    res = MonteCarloEngine(n_paths=100_000, seed=42, **reglages).simulate(option, marche)
    return res.price, res.std_error


def arbre(n):
    return BinomialTreeEngine(n_steps=n).price(option, marche), None


METHODES = [
    ("Black-Scholes (formule fermée)", lambda: (BlackScholesEngine().price(option, marche), None)),
    ("Monte Carlo simple, 100 000 tirages", lambda: monte_carlo()),
    ("Monte Carlo antithétiques + contrôle, 100 000 tirages", lambda: monte_carlo(antithetic=True, control_variate=True)),
    ("Arbre CRR, 100 pas", lambda: arbre(100)),
    ("Arbre CRR, 1 000 pas", lambda: arbre(1000)),
]

# Tableau au format Markdown : il se copie directement dans le README.
print("| Méthode | Prix | Écart à Black-Scholes | Erreur standard | Temps (ms) |")
print("|---|---:|---:|---:|---:|")
for nom, fonction in METHODES:
    (prix, erreur_std), temps = chronometrer(fonction)
    texte_erreur = f"{erreur_std:.4f}" if erreur_std is not None else "—"
    print(f"| {nom} | {prix:.4f} | {prix - prix_bs:+.4f} | {texte_erreur} | {temps:.2f} |")