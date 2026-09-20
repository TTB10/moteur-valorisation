"""Les stratégies optionnelles classiques, construites comme des portefeuilles."""

from pricer.instruments import EuropeanOption
from pricer.portfolio import Portfolio


def bull_spread(k_bas, k_haut, maturite):
    """Call acheté bas, call vendu haut : pari sur une hausse modérée, risque borné."""
    if k_bas >= k_haut:
        raise ValueError("Il faut k_bas < k_haut.")
    return Portfolio.from_pairs([
        (EuropeanOption(k_bas, maturite, "call"), +1),
        (EuropeanOption(k_haut, maturite, "call"), -1),
    ], name=f"Bull spread {k_bas}/{k_haut}")


def bear_spread(k_bas, k_haut, maturite):
    """Put acheté haut, put vendu bas : pari sur une baisse modérée."""
    if k_bas >= k_haut:
        raise ValueError("Il faut k_bas < k_haut.")
    return Portfolio.from_pairs([
        (EuropeanOption(k_haut, maturite, "put"), +1),
        (EuropeanOption(k_bas, maturite, "put"), -1),
    ], name=f"Bear spread {k_bas}/{k_haut}")


def straddle(strike, maturite):
    """Call et put au même strike : pari sur l'amplitude, pas sur la direction."""
    return Portfolio.from_pairs([
        (EuropeanOption(strike, maturite, "call"), +1),
        (EuropeanOption(strike, maturite, "put"), +1),
    ], name=f"Straddle {strike}")


def strangle(k_put, k_call, maturite):
    """Put OTM et call OTM : même pari qu'un straddle, moins cher, mais il faut plus d'amplitude."""
    if k_put >= k_call:
        raise ValueError("Il faut k_put < k_call.")
    return Portfolio.from_pairs([
        (EuropeanOption(k_put, maturite, "put"), +1),
        (EuropeanOption(k_call, maturite, "call"), +1),
    ], name=f"Strangle {k_put}/{k_call}")


def butterfly(k_bas, k_centre, k_haut, maturite):
    """Deux ailes achetées, deux corps vendus : pari sur la stabilité autour du centre."""
    if not k_bas < k_centre < k_haut:
        raise ValueError("Il faut k_bas < k_centre < k_haut.")
    return Portfolio.from_pairs([
        (EuropeanOption(k_bas, maturite, "call"), +1),
        (EuropeanOption(k_centre, maturite, "call"), -2),
        (EuropeanOption(k_haut, maturite, "call"), +1),
    ], name=f"Butterfly {k_bas}/{k_centre}/{k_haut}")