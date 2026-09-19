"""Convergence du prix Monte Carlo, avec et sans réduction de variance."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pricer.engines.analytic import BlackScholesEngine
from pricer.engines.monte_carlo import MonteCarloEngine
from pricer.instruments import EuropeanOption
from pricer.market_data import MarketData

marche = MarketData(spot=100, rate=0.05, dividend=0.0, vol=0.2)
option = EuropeanOption(strike=100, maturity=1.0)
prix_bs = BlackScholesEngine().price(option, marche)

# De 100 à 1 000 000 tirages, espacés régulièrement en échelle log, arrondis à un nombre pair.
TAILLES = np.unique(np.logspace(2, 6, 25).astype(int) // 2 * 2)

METHODES = {
    "Monte Carlo simple": {},
    "Antithétiques": {"antithetic": True},
    "Variable de contrôle": {"control_variate": True},
    "Antithétiques + contrôle": {"antithetic": True, "control_variate": True},
}

fig, (ax_prix, ax_err) = plt.subplots(1, 2, figsize=(14, 5.5))
erreurs_simple = None

for nom, reglages in METHODES.items():
    resultats = [
        MonteCarloEngine(n_paths=int(n), seed=42, **reglages).simulate(option, marche)
        for n in TAILLES
    ]
    prix = np.array([r.price for r in resultats])
    erreurs = np.array([r.std_error for r in resultats])
    if erreurs_simple is None:
        erreurs_simple = erreurs

    (ligne,) = ax_prix.plot(TAILLES, prix, marker="o", markersize=3, label=nom)
    ax_prix.fill_between(TAILLES, prix - 1.96 * erreurs, prix + 1.96 * erreurs,
                         color=ligne.get_color(), alpha=0.15)
    ax_err.plot(TAILLES, erreurs, marker="o", markersize=3, label=nom)

# Graphe de gauche : prix estimé et intervalle à 95 %
ax_prix.axhline(prix_bs, color="black", linestyle="--", linewidth=1, label="Black-Scholes")
ax_prix.set_xscale("log")
ax_prix.set_ylim(prix_bs - 2, prix_bs + 2)
ax_prix.set_xlabel("Nombre de spots simulés")
ax_prix.set_ylabel("Prix estimé")
ax_prix.set_title("Prix et intervalle de confiance à 95 %")
ax_prix.legend()

# Graphe de droite : erreur standard en log-log, avec une droite de pente -1/2 pour référence
reference = erreurs_simple[-1] * np.sqrt(TAILLES[-1] / TAILLES)
ax_err.plot(TAILLES, reference, "k:", label="Pente −1/2")
ax_err.set_xscale("log")
ax_err.set_yscale("log")
ax_err.set_xlabel("Nombre de spots simulés")
ax_err.set_ylabel("Erreur standard")
ax_err.set_title("Convergence en 1/√n")
ax_err.legend()

fig.suptitle("Call européen : S = K = 100, T = 1, r = 5 %, σ = 20 %")
fig.tight_layout()

Path("figures").mkdir(exist_ok=True)
chemin = Path("figures") / "convergence_monte_carlo.png"
fig.savefig(chemin, dpi=150)
print(f"Graphe enregistré : {chemin}")