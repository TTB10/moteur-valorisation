# Moteur de valorisation d'options

Moteur de valorisation d'options en Python, construit autour d'un principe :
aucune méthode de calcul n'est considérée comme juste tant qu'elle n'est pas
vérifiée par une méthode indépendante.

## Architecture

Le projet sépare trois notions :

- **`MarketData`** : l'état du marché (spot, taux, dividende, volatilité).
- **`Instrument`** : le contrat et ce qu'il paie à l'échéance (`EuropeanOption`).
- **`PricingEngine`** : la méthode de calcul du prix (`BlackScholesEngine`).

Un même instrument peut ainsi être valorisé par plusieurs moteurs, ce qui
permet de comparer les méthodes entre elles.

## Avancement

- [x] Moteur analytique Black-Scholes
- [ ] Monte Carlo avec réduction de variance
- [ ] Arbre binomial Cox-Ross-Rubinstein
- [ ] Grecques analytiques et par différences finies
- [ ] Couverture dynamique en delta
- [ ] Surface de volatilité implicite
- [ ] VaR et Expected Shortfall

## Installation

```bash
git clone https://github.com/TTB10/moteur-valorisation.git
cd moteur-valorisation
python -m venv .venv
pip install -r requirements.txt
```

## Utilisation

```python
from pricer.market_data import MarketData
from pricer.instruments import EuropeanOption
from pricer.engines.analytic import BlackScholesEngine

marche = MarketData(spot=100, rate=0.05, dividend=0.0, vol=0.2)
option = EuropeanOption(strike=100, maturity=1.0, option_type="call")

print(BlackScholesEngine().price(option, marche))  # 10.4506
```

## Tests

```bash
python -m pytest
```

- Parité call-put vérifiée sur 720 combinaisons de paramètres, à 1e-10 près.
- Comparaison avec une valeur de référence de la littérature.