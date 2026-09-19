from dataclasses import dataclass


@dataclass(frozen=True)
class MarketData:
    """État du marché à un instant donné."""

    spot: float      # S : prix actuel du sous-jacent
    rate: float      # r : taux sans risque annuel, composé en continu
    dividend: float  # q : taux de dividende annuel, en continu
    vol: float       # sigma : volatilité annuelle

    def __post_init__(self):
        if self.spot <= 0:
            raise ValueError("Le spot doit être strictement positif.")
        if self.vol <= 0:
            raise ValueError("La volatilité doit être strictement positive.")