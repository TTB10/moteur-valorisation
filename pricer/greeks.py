from dataclasses import dataclass


@dataclass(frozen=True)
class Greeks:
    """Sensibilités du prix, en dérivées mathématiques pures.

    Conventions :
      delta : par unité de spot
      gamma : par unité de spot au carré
      vega  : par 1.00 de volatilité   (par point de vol : vega / 100)
      theta : par an                   (par jour calendaire : theta / 365)
      rho   : par 1.00 de taux         (par point de base : rho / 10_000)
    """

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float