"""Convergence de l'arbre CRR vers Black-Scholes : oscillation et vitesse en 1/N."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pricer.engines.analytic import BlackScholesEngine
from pricer.engines.binomial import BinomialTreeEngine
from pricer.instruments import EuropeanOption
from pricer.market_data import MarketData

marche = MarketData(spot=100, rate=0.05, dividend=0.0, vol=0.2)
option = EuropeanOption(strike=100, maturity=1.0)
prix_bs = BlackScholesEngine().price(option, marche)

N_VALEURS = np.arange(10, 1001)
erreurs = np.array([BinomialTreeEngine(int(n)).price(option, marche) - prix_bs for n in N_VALEURS])
moyennes = 0.5 * (erreurs[:-1] + erreurs[1:])   # moyenne des arbres à N et N+1 pas

fig, (ax_osc, ax_log) = plt.subplots(1, 2, figsize=(14, 5.5))

# Graphe de gauche : l'erreur signée, pour voir l'oscillation pair / impair
ax_osc.plot(N_VALEURS, erreurs, linewidth=0.8, label="Arbre CRR")
ax_osc.plot(N_VALEURS[:-1], moyennes, linewidth=1.5, label="Moyenne des arbres à N et N+1 pas")
ax_osc.axhline(0, color="black", linestyle="--", linewidth=1)
ax_osc.set_xlim(10, 200)
ax_osc.set_xlabel("Nombre de pas N")
ax_osc.set_ylabel("Prix de l'arbre − prix Black-Scholes")
ax_osc.set_title("Oscillation de l'erreur")
ax_osc.legend()

# Graphe de droite : l'erreur absolue en log-log, avec une droite de pente -1 pour référence
reference = abs(erreurs[-1]) * N_VALEURS[-1] / N_VALEURS
ax_log.loglog(N_VALEURS, np.abs(erreurs), linewidth=0.8, label="Arbre CRR")
ax_log.loglog(N_VALEURS[:-1], np.abs(moyennes), linewidth=0.8, label="Moyenne N et N+1")
ax_log.loglog(N_VALEURS, reference, "k:", label="Pente −1")
ax_log.set_xlabel("Nombre de pas N")
ax_log.set_ylabel("|Erreur|")
ax_log.set_title("Convergence en 1/N")
ax_log.legend()

fig.suptitle("Call européen : S = K = 100, T = 1, r = 5 %, σ = 20 %")
fig.tight_layout()

Path("figures").mkdir(exist_ok=True)
chemin = Path("figures") / "convergence_arbre.png"
fig.savefig(chemin, dpi=150)
print(f"Graphe enregistré : {chemin}")