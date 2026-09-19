"""Compare les techniques de réduction de variance à nombre égal de spots simulés."""

from pricer.engines.analytic import BlackScholesEngine
from pricer.engines.monte_carlo import MonteCarloEngine
from pricer.instruments import EuropeanOption
from pricer.market_data import MarketData

marche = MarketData(spot=100, rate=0.05, dividend=0.0, vol=0.2)
option = EuropeanOption(strike=100, maturity=1.0)
N = 100_000

METHODES = {
    "Monte Carlo simple": {},
    "Antithétiques": {"antithetic": True},
    "Variable de contrôle": {"control_variate": True},
    "Antithétiques + contrôle": {"antithetic": True, "control_variate": True},
}

print(f"Prix Black-Scholes : {BlackScholesEngine().price(option, marche):.4f}\n")
print(f"{'Méthode':<28}{'Prix':>10}{'Err. std':>10}{'Gain':>8}")

erreur_reference = None
for nom, reglages in METHODES.items():
    res = MonteCarloEngine(n_paths=N, seed=42, **reglages).simulate(option, marche)
    if erreur_reference is None:
        erreur_reference = res.std_error
    gain = (erreur_reference / res.std_error) ** 2
    print(f"{nom:<28}{res.price:>10.4f}{res.std_error:>10.4f}{gain:>8.1f}")