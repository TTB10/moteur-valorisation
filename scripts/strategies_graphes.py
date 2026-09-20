"""Profils de gain et grecques agrégées des cinq stratégies classiques."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pricer.market_data import MarketData
from pricer.strategies import bear_spread, bull_spread, butterfly, straddle, strangle

MARCHE = MarketData(spot=100, rate=0.05, dividend=0.0, vol=0.2)
T = 1.0
SPOTS = np.linspace(60, 140, 400)
GRECQUES = ("delta", "gamma", "vega", "theta")

STRATEGIES = [
    bull_spread(95, 105, T),
    bear_spread(95, 105, T),
    straddle(100, T),
    strangle(90, 110, T),
    butterfly(90, 100, 110, T),
]


def valeurs_contre_spot(strategie, fonction):
    """Applique une fonction du marché décalé, pour chaque spot de la grille."""
    from dataclasses import replace
    return np.array([fonction(strategie, replace(MARCHE, spot=S)) for S in SPOTS])


# --- Figure 1 : profils de gain (payoff et P&L net de la prime)
fig1, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
for ax, s in zip(axes.flat, STRATEGIES):
    prime = s.price(MARCHE)
    ax.plot(SPOTS, s.payoff(SPOTS), linewidth=1.8, label="Payoff à maturité")
    ax.plot(SPOTS, s.payoff(SPOTS) - prime * np.exp(MARCHE.rate * T), "--",
            linewidth=1.3, label="P&L net de la prime capitalisée")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(MARCHE.spot, color="grey", linestyle=":", linewidth=0.8)
    ax.set_title(f"{s.name}\nprime = {prime:.2f}", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
axes.flat[-1].axis("off")
for ax in axes[1]:
    ax.set_xlabel("Spot à maturité")
fig1.suptitle("Profils de gain des cinq stratégies (T = 1 an, S₀ = 100)")
fig1.tight_layout()

# --- Figure 2 : grecques agrégées contre le spot
fig2, axes = plt.subplots(2, 2, figsize=(14, 9))
for ax, nom in zip(axes.flat, GRECQUES):
    for s in STRATEGIES:
        valeurs = valeurs_contre_spot(s, lambda st, m, n=nom: getattr(st.greeks(m), n))
        ax.plot(SPOTS, valeurs, linewidth=1.4, label=s.name)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(100, color="grey", linestyle=":", linewidth=0.8)
    ax.set_title(nom.capitalize())
    ax.set_xlabel("Spot")
    ax.grid(alpha=0.3)
axes.flat[0].legend(fontsize=8)
fig2.suptitle("Grecques agrégées des stratégies (T = 1 an)")
fig2.tight_layout()

dossier = Path("figures")
dossier.mkdir(exist_ok=True)
for nom_fichier, figure in (("strategies_payoffs.png", fig1),
                            ("strategies_grecques.png", fig2)):
    figure.savefig(dossier / nom_fichier, dpi=150)
    print(f"Graphe enregistré : figures/{nom_fichier}")

# --- Tableau récapitulatif au format Markdown
print(f"\n| Stratégie | Prime | Delta | Gamma | Vega | Thêta | Pari |")
print("|---|---:|---:|---:|---:|---:|---|")
PARIS = ["Hausse modérée", "Baisse modérée", "Forte amplitude",
         "Forte amplitude (moins cher)", "Stabilité"]
for s, pari in zip(STRATEGIES, PARIS):
    g = s.greeks(MARCHE)
    print(f"| {s.name} | {s.price(MARCHE):.2f} | {g.delta:+.3f} | {g.gamma:+.4f} "
          f"| {g.vega:+.1f} | {g.theta:+.2f} | {pari} |")