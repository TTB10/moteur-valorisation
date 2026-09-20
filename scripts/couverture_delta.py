"""Distribution de l'erreur de couverture en delta selon la fréquence de rebalancement."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pricer.hedging import simulate_delta_hedge
from pricer.instruments import EuropeanOption
from pricer.market_data import MarketData

MARCHE = MarketData(spot=100, rate=0.05, dividend=0.0, vol=0.2)
CALL = EuropeanOption(strike=100, maturity=1.0, option_type="call")
N_PATHS = 50_000
GRAINE = 2026

FREQUENCES = {
    "Mensuel (N = 12)": 12,
    "Hebdomadaire (N = 52)": 52,
    "Quotidien (N = 252)": 252,
    "4 fois par jour (N = 1008)": 1008,
}

resultats = {nom: simulate_delta_hedge(CALL, MARCHE, n, n_paths=N_PATHS, seed=GRAINE)
             for nom, n in FREQUENCES.items()}
prime = next(iter(resultats.values())).premium

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
ax_dist, ax_conv, ax_quant, ax_vol = axes.flat

# --- 1. Distributions superposées
for nom, res in resultats.items():
    ax_dist.hist(res.errors, bins=200, range=(-3, 3), density=True,
                 histtype="step", linewidth=1.5, label=f"{nom} : σ = {res.std:.3f}")
ax_dist.axvline(0, color="black", linewidth=0.8)
ax_dist.set_xlabel("Erreur de couverture (prime = %.2f)" % prime)
ax_dist.set_ylabel("Densité")
ax_dist.set_title("Distribution de l'erreur selon la fréquence")
ax_dist.legend(fontsize=9)
ax_dist.grid(alpha=0.3)

# --- 2. Convergence de l'écart-type en 1/racine(N)
N_GRILLE = np.array([4, 8, 16, 32, 63, 126, 252, 504, 1008])
ecarts = np.array([simulate_delta_hedge(CALL, MARCHE, int(n), n_paths=20_000, seed=GRAINE).std
                   for n in N_GRILLE])
reference = ecarts[-1] * np.sqrt(N_GRILLE[-1] / N_GRILLE)

ax_conv.loglog(N_GRILLE, ecarts, "o-", markersize=4, label="Écart-type mesuré")
ax_conv.loglog(N_GRILLE, reference, "k:", label="Pente −1/2")
ax_conv.set_xlabel("Nombre de rebalancements N")
ax_conv.set_ylabel("Écart-type de l'erreur")
ax_conv.set_title("L'erreur décroît en 1/√N : elle ne disparaît jamais")
ax_conv.legend()
ax_conv.grid(alpha=0.3, which="both")

# --- 3. Moyenne et quantiles extrêmes
noms = list(resultats)
x = np.arange(len(noms))
moyennes = [resultats[n].mean for n in noms]
q01 = [resultats[n].quantile(0.01) for n in noms]
q99 = [resultats[n].quantile(0.99) for n in noms]

ax_quant.plot(x, moyennes, "o-", label="Moyenne")
ax_quant.plot(x, q01, "v--", label="Quantile 1 % (perte extrême)")
ax_quant.plot(x, q99, "^--", label="Quantile 99 % (gain extrême)")
ax_quant.axhline(0, color="black", linewidth=0.8)
ax_quant.set_xticks(x)
ax_quant.set_xticklabels([n.split(" (")[0] for n in noms], rotation=15)
ax_quant.set_ylabel("Erreur")
ax_quant.set_title("Moyenne nulle, mais queue de perte plus lourde")
ax_quant.legend(fontsize=9)
ax_quant.grid(alpha=0.3)

# --- 4. Couvrir avec une volatilité différente de la volatilité réalisée
VOLS_REALISEES = np.linspace(0.10, 0.30, 11)
VOL_COUVERTURE = 0.20
pnl_moyen, pnl_ecart = [], []
for vol_reelle in VOLS_REALISEES:
    res = simulate_delta_hedge(CALL, MarketData(spot=100, rate=0.05, dividend=0.0, vol=vol_reelle),
                               252, n_paths=20_000, seed=GRAINE, hedge_vol=VOL_COUVERTURE)
    pnl_moyen.append(res.mean)
    pnl_ecart.append(res.std)

pnl_moyen, pnl_ecart = np.array(pnl_moyen), np.array(pnl_ecart)
ax_vol.plot(VOLS_REALISEES * 100, pnl_moyen, "o-", label="P&L moyen du vendeur")
ax_vol.fill_between(VOLS_REALISEES * 100, pnl_moyen - pnl_ecart, pnl_moyen + pnl_ecart,
                    alpha=0.2, label="± 1 écart-type")
ax_vol.axhline(0, color="black", linewidth=0.8)
ax_vol.axvline(VOL_COUVERTURE * 100, color="grey", linestyle=":",
               label="Volatilité de couverture (20 %)")
ax_vol.set_xlabel("Volatilité réalisée (%)")
ax_vol.set_ylabel("P&L du vendeur")
ax_vol.set_title("Vendre à 20 % : on gagne si le marché bouge moins")
ax_vol.legend(fontsize=9)
ax_vol.grid(alpha=0.3)

fig.suptitle("Couverture dynamique en delta d'un call vendu "
             "(S = K = 100, T = 1 an, r = 5 %, σ = 20 %)")
fig.tight_layout()

Path("figures").mkdir(exist_ok=True)
chemin = Path("figures") / "couverture_delta.png"
fig.savefig(chemin, dpi=150)
print(f"Graphe enregistré : {chemin}")

print(f"\nPrime encaissée : {prime:.4f}")
print(f"{'Fréquence':<28}{'Moyenne':>10}{'Écart-type':>12}{'Q1 %':>10}{'Q99 %':>10}")
for nom, res in resultats.items():
    print(f"{nom:<28}{res.mean:>10.4f}{res.std:>12.4f}"
          f"{res.quantile(0.01):>10.3f}{res.quantile(0.99):>10.3f}")