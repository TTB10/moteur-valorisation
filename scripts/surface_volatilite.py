"""Construit et trace la surface de volatilité implicite sur données de marché réelles."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from pricer.market_data_feed import clean_option_chain, fetch_option_chain
from pricer.vol_surface import (build_vol_surface, butterfly_arbitrage_violations,
                                calendar_arbitrage_violations)

parser = argparse.ArgumentParser()
parser.add_argument("--ticker", default="SPY")
parser.add_argument("--rate", type=float, default=0.04, help="taux sans risque annuel continu")
args = parser.parse_args()


brut, spot, horodatage = fetch_option_chain(args.ticker)
propre, journal_nettoyage = clean_option_chain(brut, spot)
# Pour la parité, il faut les strikes cotés en call ET en put : donc sans le filtre OTM.
avec_itm, _ = clean_option_chain(brut, spot, garder_otm_seulement=False)
surface, journal_surface = build_vol_surface(propre, spot, args.rate,
                                             propre_pour_parite=avec_itm)

print(f"{args.ticker} — spot {spot:.2f} — capture {horodatage:%Y-%m-%d %H:%M UTC}\n")
print("Nettoyage des cotations")
for k, v in journal_nettoyage.items():
    print(f"  {k:<42} {v}")
print("\nInversion de Black-Scholes")
for k, v in journal_surface.items():
    print(f"  {k:<42} {v}")

print("\nStructure par terme à la monnaie")
print(surface.atm_term_structure().to_string(index=False,
      formatters={"maturity": "{:.3f}".format, "forward": "{:.2f}".format,
                  "iv_atm": "{:.2%}".format}))

print("\nCourbe des forwards : régression de ln(F) sur T")
T_obs = np.array(surface.maturities)
lnF = np.array([np.log(surface.forwards[T]) for T in T_obs])
pente, ordonnee = np.polyfit(T_obs, lnF, 1)
spot_implicite = np.exp(ordonnee)
residus = lnF - (pente * T_obs + ordonnee)

print(f"  taux de portage implicite r - q  {pente:6.2%}")
print(f"  spot implicite par les options   {spot_implicite:7.2f}"
      f"   (spot coté {spot:.2f}, écart {spot_implicite - spot:+.2f})")
print(f"  dividende implicite si r = {args.rate:.1%}   {args.rate - pente:6.2%}")
print(f"  écart-type des résidus           {residus.std():.5f}")
print("\nTests d'absence d'arbitrage")
total_papillon = sum(butterfly_arbitrage_violations(surface, T) for T in surface.maturities)


print(f"  violations papillon (convexité en strike)  {total_papillon}")
print(f"  violations calendaires (variance totale)   {calendar_arbitrage_violations(surface)}")

# --- Figure 1 : les smiles, une courbe par maturité
fig1, (ax_k, ax_m) = plt.subplots(1, 2, figsize=(14, 5.5))
couleurs = cm.viridis(np.linspace(0, 0.9, len(surface.maturities)))

for T, couleur in zip(surface.maturities, couleurs):
    tranche = surface.smile(T)
    ax_k.plot(tranche["strike"], tranche["iv"] * 100, "o-", markersize=3,
              color=couleur, linewidth=1.2, label=f"T = {T:.2f} an")
    ax_m.plot(tranche["log_moneyness"], tranche["iv"] * 100, "o-", markersize=3,
              color=couleur, linewidth=1.2, label=f"T = {T:.2f} an")

ax_k.axvline(spot, color="grey", linestyle=":", linewidth=1)
ax_k.set_xlabel("Strike")
ax_k.set_ylabel("Volatilité implicite (%)")
ax_k.set_title("Smile par strike")
ax_k.grid(alpha=0.3)
ax_k.legend(fontsize=8)

ax_m.axvline(0, color="grey", linestyle=":", linewidth=1)
ax_m.set_xlabel("Log-moneyness  ln(K / F)")
ax_m.set_ylabel("Volatilité implicite (%)")
ax_m.set_title("Smile par moneyness : les maturités deviennent comparables")
ax_m.grid(alpha=0.3)

fig1.suptitle(f"{args.ticker} — smile de volatilité implicite "
              f"(spot {spot:.2f}, {horodatage:%Y-%m-%d %H:%M UTC})")
fig1.tight_layout()

# --- Figure 2 : la surface en trois dimensions
fig2 = plt.figure(figsize=(13, 6))

ax3d = fig2.add_subplot(1, 2, 1, projection="3d")
pts = surface.points
ax3d.plot_trisurf(pts["log_moneyness"], pts["maturity"], pts["iv"] * 100,
                  cmap="viridis", alpha=0.85, linewidth=0.2, edgecolor="grey")
ax3d.set_xlabel("ln(K / F)")
ax3d.set_ylabel("Maturité (années)")
ax3d.set_zlabel("Vol. implicite (%)")
ax3d.set_title("Surface de volatilité implicite")
ax3d.view_init(elev=22, azim=-125)

ax_atm = fig2.add_subplot(1, 2, 2)
atm = surface.atm_term_structure()
ax_atm.plot(atm["maturity"], atm["iv_atm"] * 100, "o-", linewidth=1.5)
ax_atm.set_xlabel("Maturité (années)")
ax_atm.set_ylabel("Volatilité implicite à la monnaie (%)")
ax_atm.set_title("Structure par terme à la monnaie")
ax_atm.grid(alpha=0.3)

fig2.suptitle(f"{args.ticker} — la volatilité n'est constante ni en strike ni en maturité")
fig2.tight_layout()

dossier = Path("figures")
dossier.mkdir(exist_ok=True)
for nom, figure in (("smile_volatilite.png", fig1), ("surface_volatilite.png", fig2)):
    figure.savefig(dossier / nom, dpi=150)
    print(f"\nGraphe enregistré : figures/{nom}")

surface.points.to_csv(dossier / "surface_points.csv", index=False)
print("Points exportés : figures/surface_points.csv")