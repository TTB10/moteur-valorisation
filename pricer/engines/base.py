from abc import ABC, abstractmethod


class PricingEngine(ABC):
    """Une méthode de calcul : elle valorise un instrument dans un marché donné."""

    @abstractmethod
    def price(self, instrument, market):
        """Renvoie le prix de l'instrument aujourd'hui."""