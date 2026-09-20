"""Portefeuille : agrégation pondérée d'instruments, avec payoff et grecques agrégés."""

from dataclasses import dataclass, field, replace

import numpy as np

from pricer.engines.analytic import BlackScholesEngine
from pricer.greeks import Greeks
from pricer.instruments import Instrument, Underlying

NOMS_GRECQUES = ("delta", "gamma", "vega", "theta", "rho")

# Grecques du sous-jacent : une action a un delta de 1, tout le reste est nul.
GRECQUES_SOUS_JACENT = Greeks(delta=1.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)


@dataclass(frozen=True)
class Position:
    """Une ligne du portefeuille : un instrument et une quantité signée."""

    instrument: Instrument
    quantity: float = 1.0


@dataclass(frozen=True)
class Portfolio(Instrument):
    """Un portefeuille est lui-même un instrument : il a un payoff (patron Composite)."""

    positions: tuple = field(default_factory=tuple)
    name: str = "Portefeuille"

    @classmethod
    def from_pairs(cls, pairs, name="Portefeuille"):
        """Construit depuis une liste de couples (instrument, quantité)."""
        return cls(tuple(Position(i, q) for i, q in pairs), name)

    def payoff(self, spot):
        spot = np.asarray(spot, dtype=float)
        total = np.zeros_like(spot)
        for p in self.positions:
            total = total + p.quantity * p.instrument.payoff(spot)
        return total

    def price(self, market, engine=None):
        """Valeur du portefeuille : somme pondérée des valeurs de chaque ligne."""
        engine = engine or BlackScholesEngine()
        total = 0.0
        for p in self.positions:
            if isinstance(p.instrument, Underlying):
                valeur = market.spot
            elif isinstance(p.instrument, Portfolio):
                valeur = p.instrument.price(market, engine)
            else:
                valeur = engine.price(p.instrument, market)
            total += p.quantity * valeur
        return float(total)

    def greeks(self, market, engine=None):
        """Grecques agrégées : la dérivation est linéaire, donc les grecques s'additionnent."""
        engine = engine or BlackScholesEngine()
        totaux = dict.fromkeys(NOMS_GRECQUES, 0.0)

        for p in self.positions:
            if isinstance(p.instrument, Underlying):
                g = GRECQUES_SOUS_JACENT
            elif isinstance(p.instrument, Portfolio):
                g = p.instrument.greeks(market, engine)
            else:
                g = engine.greeks(p.instrument, market)
            for nom in NOMS_GRECQUES:
                totaux[nom] += p.quantity * getattr(g, nom)

        return Greeks(**totaux)

    def delta_hedged(self, market, engine=None):
        """Renvoie une copie neutralisée en delta par une position sur le sous-jacent."""
        delta = self.greeks(market, engine).delta
        positions = self.positions + (Position(Underlying(), -delta),)
        return replace(self, positions=positions, name=f"{self.name} (delta-neutre)")

    def __len__(self):
        return len(self.positions)