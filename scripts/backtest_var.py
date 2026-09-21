"""Backtest de trois modèles de VaR sur l'historique réel d'un sous-jacent."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pricer.backtest import basel_zone, exceptions, kupiec_test, rolling_var_forecasts
from pricer.market_data_feed import fetch_price_history

parser = argparse.ArgumentParser()
parser.add_argument("--ticker", default="SPY")
parser.add_argument("--period", default="10y")
parser.add_argument("--confidence", type=float, default=0.99)
args = parser.parse_args()

prix = fetch_price_history(args.ticker, args.period)
rendements = np.log(prix).diff().dropna()
previsions = rolling_var_forecasts(rendements, args.confidence)

METHODES = {
    "historique": "Historique (fenêtre 250 j)",
    "normale": "Normale (fenêtre 250 j)",
    "ewma": "Normale EWMA (λ = 0,94)",
}

debut, fin = previsions.index[0].date(), previsions.index[-1].date()
print(f"{args.ticker} — VaR {args.confidence:.0%} à 1 jour — {len(previsions)} jours "
      f"du {debut} au {fin}\n")
print(f"{'Modèle':<30}{'Exc.':>6}{'Attendu':>9}{'Taux':>8}{'LR':>8}{'p-value':>9}"
      f"{'Kupiec':>10}{'Bâle 250 j':>12}")

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
for ax, (col, nom) in zip(axes, METHODES.items()):
    exc = exceptions(previsions, col)
    test = kupiec_test(int(exc.sum()), len(previsions), args.confidence)
    zone = basel_zone(int(exc.iloc[-250:].sum()))
    verdict = "rejeté" if test.rejected else "accepté"
    print(f"{nom:<30}{test.n_exceptions:>6}{test.expected:>9.1f}{test.rate:>8.2%}"
          f"{test.lr:>8.2f}{test.p_value:>9.3f}{verdict:>10}{zone:>12}")

    ax.plot(previsions.index, previsions["perte"] * 100, color="grey", linewidth=0.5,
            label="Perte réalisée")
    ax.plot(previsions.index, previsions[col] * 100, linewidth=1.2, label=f"VaR {nom}")
    ax.scatter(previsions.index[exc], previsions["perte"][exc] * 100, color="red", s=10,
               zorder=3, label=f"Exceptions : {test.n_exceptions} (attendu {test.expected:.0f})")
    ax.set_ylabel("Perte (%)")
    ax.set_title(f"{nom} — Kupiec {verdict} (LR = {test.lr:.2f})")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

fig.suptitle(f"{args.ticker} — backtest de la VaR à {args.confidence:.0%} à un jour")
fig.tight_layout()

Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/backtest_var.png", dpi=150)
print("\nGraphe enregistré : figures/backtest_var.png")

print("\nLes cinq pires journées")
pires = previsions.nlargest(5, "perte")
for date, ligne in pires.iterrows():
    print(f"  {date.date()}  perte {ligne['perte']:6.2%}   VaR historique {ligne['historique']:6.2%}"
          f"   VaR normale {ligne['normale']:6.2%}   VaR EWMA {ligne['ewma']:6.2%}")