"""Produit les quatre figures de la bibliothèque de graphes."""

from pathlib import Path

from pricer.market_data import MarketData
from pricer.plots import (grecques_contre_spot, grecques_contre_temps,
                          payoffs_quatre_positions, prix_contre_spot)

marche = MarketData(spot=100, rate=0.05, dividend=0.0, vol=0.2)

FIGURES = {
    "payoffs.png": payoffs_quatre_positions(),
    "prix_contre_spot.png": prix_contre_spot(marche),
    "grecques_contre_spot.png": grecques_contre_spot(marche),
    "grecques_contre_temps.png": grecques_contre_temps(marche),
}

dossier = Path("figures")
dossier.mkdir(exist_ok=True)
for nom, figure in FIGURES.items():
    chemin = dossier / nom
    figure.savefig(chemin, dpi=150)
    print(f"Graphe enregistré : {chemin}")