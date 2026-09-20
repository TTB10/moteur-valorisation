"""Construction de la surface de volatilité implicite à partir de prix cotés."""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from pricer.engines.analytic import BlackScholesEngine
from pricer.implied_vol import NoImpliedVolatility, implied_volatility
from pricer.instruments import EuropeanOption
from pricer.market_data import MarketData
from pricer.market_data_feed import implied_forward_by_parity

MOTEUR = BlackScholesEngine()

VOL_PLANCHER = 0.01     # 1 %
VOL_PLAFOND = 3.0       # 300 %
VEGA_MIN_SURFACE = 0.5  # en dessous, la vol implicite est trop imprécise (Δσ ≈ spread / vega)


@dataclass(frozen=True)
class VolSurface:
    """Points de volatilité implicite, avec le forward estimé par maturité."""

    points: pd.DataFrame          # colonnes : maturity, strike, type, iv, vega, log_moneyness...
    forwards: dict                # maturité -> forward implicite
    spot: float
    rate: float

    @property
    def maturities(self):
        return sorted(self.forwards)

    def smile(self, maturite):
        """Les points d'une maturité donnée, triés par strike."""
        tranche = self.points[np.isclose(self.points["maturity"], maturite)]
        return tranche.sort_values("strike")

    def atm_term_structure(self):
        """Volatilité à la monnaie (au forward) par maturité, par interpolation."""
        lignes = []
        for T in self.maturities:
            tranche = self.smile(T)
            if len(tranche) < 2:
                continue
            F = self.forwards[T]
            iv_atm = np.interp(F, tranche["strike"].values, tranche["iv"].values)
            lignes.append({"maturity": T, "forward": F, "iv_atm": float(iv_atm)})
        return pd.DataFrame(lignes)

    def total_variance(self):
        """Variance totale w = sigma^2 T, utile pour tester l'arbitrage calendaire."""
        df = self.points.copy()
        df["total_variance"] = df["iv"] ** 2 * df["maturity"]
        return df


def build_vol_surface(propre, spot, rate, propre_pour_parite=None):
    """Inverse Black-Scholes sur chaque cotation retenue. Renvoie (VolSurface, journal).

    propre_pour_parite : chaîne nettoyée SANS le filtre OTM, nécessaire pour estimer le
    forward par parité (elle seule contient des strikes cotés à la fois en call et en put).
    Si elle est absente, on retombe sur F = S e^{rT}, ce qui suppose un dividende nul.
    """
    source_parite = propre if propre_pour_parite is None else propre_pour_parite
    journal = {"cotations en entrée": len(propre)}
    forwards, lignes = {}, []
    echecs_inversion = vega_faible = hors_bornes_vol = 0
    forwards_par_repli = 0

    for maturite in sorted(propre["maturity"].unique()):
        tranche = propre[np.isclose(propre["maturity"], maturite)]

        forward = implied_forward_by_parity(source_parite, maturite, rate)
        if forward is None:
            forward = spot * np.exp(rate * maturite)
            forwards_par_repli += 1
        forwards[maturite] = forward

        # q tel que F = S e^{(r-q)T} : le dividende implicite révélé par la parité.
        dividende = rate - np.log(forward / spot) / maturite
        marche = MarketData(spot=spot, rate=rate, dividend=dividende, vol=0.3)

        for _, ligne in tranche.iterrows():
            option = EuropeanOption(float(ligne["strike"]), float(maturite), ligne["type"])
            try:
                iv = implied_volatility(float(ligne["mid"]), option, marche)
            except NoImpliedVolatility:
                echecs_inversion += 1
                continue

            if not (VOL_PLANCHER < iv < VOL_PLAFOND):
                hors_bornes_vol += 1
                continue

            vega = MOTEUR.greeks(option, replace(marche, vol=iv)).vega
            if vega < VEGA_MIN_SURFACE:
                vega_faible += 1
                continue

            lignes.append({
                "maturity": float(maturite),
                "strike": float(ligne["strike"]),
                "type": ligne["type"],
                "mid": float(ligne["mid"]),
                "iv": float(iv),
                "vega": float(vega),
                "forward": forward,
                "log_moneyness": float(np.log(ligne["strike"] / forward)),
                "spread_relatif": float(ligne.get("spread_relatif", np.nan)),
            })

    journal.update({
        "forwards estimés par repli (sans parité)": forwards_par_repli,
        "inversion impossible (bornes d'arbitrage)": echecs_inversion,
        "volatilité hors plage plausible": hors_bornes_vol,
        "vega trop faible pour être fiable": vega_faible,
        "points retenus": len(lignes),
    })

    return VolSurface(pd.DataFrame(lignes), forwards, spot, rate), journal


def butterfly_arbitrage_violations(surface, maturite, tolerance=1e-4):
    """Viole-t-on la convexité du prix du call en strike (densité risque-neutre négative) ?

    Pour trois strikes K1 < K2 < K3 non nécessairement équidistants, la convexité s'écrit
    C(K2) <= w C(K1) + (1-w) C(K3) avec w = (K3-K2)/(K3-K1). La seconde différence brute
    n'est valable que pour des strikes équidistants.
    """
    tranche = surface.smile(maturite)
    calls = tranche[tranche["type"] == "call"].sort_values("strike")
    if len(calls) < 3:
        return 0

    K = calls["strike"].values
    C = calls["mid"].values
    poids = (K[2:] - K[1:-1]) / (K[2:] - K[:-2])
    interpolation = poids * C[:-2] + (1 - poids) * C[2:]

    # Un arbitrage n'est réel que s'il survit aux coûts d'exécution : on vend le corps au bid
    # et on achète les ailes à l'ask. La marge exigée est donc la somme des demi-spreads.
    demi_spread = 0.5 * calls["mid"].values * calls["spread_relatif"].fillna(0).values
    marge = demi_spread[1:-1] + poids * demi_spread[:-2] + (1 - poids) * demi_spread[2:]

    return int((C[1:-1] > interpolation + marge + tolerance).sum())




def calendar_arbitrage_violations(surface, tolerance=1e-6):
    """Compte les cas où la variance totale décroît avec la maturité, à moneyness fixée."""
    df = surface.total_variance()
    if df.empty:
        return 0

    violations = 0
    grille_moneyness = np.linspace(-0.2, 0.2, 9)
    maturites = surface.maturities

    for m in grille_moneyness:
        variances = []
        for T in maturites:
            tranche = df[np.isclose(df["maturity"], T)].sort_values("log_moneyness")
            if len(tranche) < 2 or not (tranche["log_moneyness"].min() <= m <= tranche["log_moneyness"].max()):
                continue
            w = np.interp(m, tranche["log_moneyness"].values, tranche["total_variance"].values)
            variances.append(w)
        violations += int((np.diff(variances) < -tolerance).sum())

    return violations