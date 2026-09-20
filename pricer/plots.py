"""Bibliothèque de graphes : payoffs, prix et grecques."""

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from pricer.engines.analytic import BlackScholesEngine
from pricer.instruments import EuropeanOption

MOTEUR = BlackScholesEngine()
GRECQUES = ("delta", "gamma", "vega", "theta", "rho")


def _prix(option, marche, spots):
    """Prix de l'option pour une grille de spots."""
    return np.array([MOTEUR.price(option, replace(marche, spot=S)) for S in spots])


def _grecque(option, marche, spots, nom):
    """Une grecque donnée pour une grille de spots."""
    return np.array([getattr(MOTEUR.greeks(option, replace(marche, spot=S)), nom)
                     for S in spots])


def payoffs_quatre_positions(strike=100, spots=None):
    """Les quatre positions élémentaires : payoff à maturité."""
    spots = np.linspace(50, 150, 400) if spots is None else spots
    call = EuropeanOption(strike, 1.0, "call")
    put = EuropeanOption(strike, 1.0, "put")

    positions = [
        ("Call acheté", call.payoff(spots)),
        ("Call vendu", -call.payoff(spots)),
        ("Put acheté", put.payoff(spots)),
        ("Put vendu", -put.payoff(spots)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, (nom, payoff) in zip(axes.flat, positions):
        ax.plot(spots, payoff, linewidth=1.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(strike, color="grey", linestyle=":", linewidth=0.8)
        ax.set_title(nom)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("Spot à maturité")
    for ax in axes[:, 0]:
        ax.set_ylabel("Payoff")

    fig.suptitle(f"Payoffs des quatre positions élémentaires (K = {strike})")
    fig.tight_layout()
    return fig


def prix_contre_spot(marche, strike=100, maturites=(0.01, 0.25, 1.0, 2.0), spots=None):
    """Valeur de l'option contre le spot, à plusieurs maturités."""
    spots = np.linspace(50, 150, 300) if spots is None else spots
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, type_option in zip(axes, ("call", "put")):
        for T in maturites:
            option = EuropeanOption(strike, T, type_option)
            ax.plot(spots, _prix(option, marche, spots), linewidth=1.5, label=f"T = {T} an")
        # Valeur intrinsèque : la limite quand T tend vers 0
        intrinseque = EuropeanOption(strike, 1.0, type_option).payoff(spots)
        ax.plot(spots, intrinseque, "k--", linewidth=1, label="Valeur intrinsèque")
        ax.axvline(strike, color="grey", linestyle=":", linewidth=0.8)
        ax.set_title(f"Prix du {type_option}")
        ax.set_xlabel("Spot")
        ax.set_ylabel("Prix")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle("Valeur de l'option contre le spot : l'écart à l'intrinsèque est la valeur temps")
    fig.tight_layout()
    return fig


def grecques_contre_spot(marche, strike=100, maturite=1.0, spots=None):
    """Les cinq grecques contre le spot, call et put."""
    spots = np.linspace(50, 150, 300) if spots is None else spots
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for ax, nom in zip(axes.flat, GRECQUES):
        for type_option in ("call", "put"):
            option = EuropeanOption(strike, maturite, type_option)
            ax.plot(spots, _grecque(option, marche, spots, nom),
                    linewidth=1.5, label=type_option)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(strike, color="grey", linestyle=":", linewidth=0.8)
        ax.set_title(nom.capitalize())
        ax.set_xlabel("Spot")
        ax.grid(alpha=0.3)
        ax.legend()

    axes.flat[-1].axis("off")   # cinq grecques dans une grille de six
    fig.suptitle(f"Grecques contre le spot (K = {strike}, T = {maturite} an)")
    fig.tight_layout()
    return fig


def grecques_contre_temps(marche, strike=100, moneyness=(90, 100, 110), maturites=None):
    """Les cinq grecques contre le temps restant, pour trois niveaux de spot."""
    maturites = np.linspace(2.0, 0.01, 300) if maturites is None else maturites
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for ax, nom in zip(axes.flat, GRECQUES):
        for S in moneyness:
            marche_S = replace(marche, spot=S)
            valeurs = [getattr(MOTEUR.greeks(EuropeanOption(strike, T, "call"), marche_S), nom)
                       for T in maturites]
            ax.plot(maturites, valeurs, linewidth=1.5, label=f"S = {S}")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.invert_xaxis()       # le temps s'écoule vers la droite
        ax.set_title(nom.capitalize())
        ax.set_xlabel("Maturité restante (années)")
        ax.grid(alpha=0.3)
        ax.legend()

    axes.flat[-1].axis("off")
    fig.suptitle(f"Grecques du call contre le temps restant (K = {strike})")
    fig.tight_layout()
    return fig