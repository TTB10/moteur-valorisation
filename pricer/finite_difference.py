"""Grecques par différences finies, applicables à n'importe quel moteur."""

from dataclasses import replace

from pricer.greeks import Greeks

# Pas par défaut, compromis troncature / arrondi.
# Dérivée première, schéma centré : erreur ~ h^2 V'''/6 + eps*V/h, optimum ~ eps^(1/3).
# Dérivée seconde : erreur ~ h^2 V''''/12 + 4 eps*V/h^2, optimum ~ eps^(1/4),
# soit un pas plus grand en absolu que l'optimum théorique de la dérivée première.
# Sur un moteur bruité (Monte Carlo) ou discret (arbre), il faut au contraire
# AGRANDIR ces pas : le bruit joue le rôle d'un epsilon effectif bien supérieur.
PAS_SPOT = 1e-4         # relatif au spot, pour le delta
PAS_SPOT_GAMMA = 1e-3   # relatif au spot, pour le gamma
PAS_VOL = 1e-4          # absolu, en volatilité
PAS_TAUX = 1e-4         # absolu, en taux
PAS_TEMPS = 1e-4        # absolu, en années


def finite_difference_greeks(engine, instrument, market,
                             pas_spot=PAS_SPOT, pas_spot_gamma=PAS_SPOT_GAMMA,
                             pas_vol=PAS_VOL, pas_taux=PAS_TAUX, pas_temps=PAS_TEMPS):
    """Les cinq grecques par schémas centrés, mêmes conventions que Greeks."""
    S, T = market.spot, instrument.maturity
    if pas_temps >= T:
        raise ValueError("Le pas de temps doit être inférieur à la maturité.")

    prix = lambda m=market, i=instrument: engine.price(i, m)
    marche_decale = lambda **kwargs: replace(market, **kwargs)
    option_decalee = lambda **kwargs: replace(instrument, **kwargs)

    V = prix()

    h = S * pas_spot
    delta = (prix(marche_decale(spot=S + h)) - prix(marche_decale(spot=S - h))) / (2 * h)

    hg = S * pas_spot_gamma
    gamma = (prix(marche_decale(spot=S + hg))
             - 2 * V
             + prix(marche_decale(spot=S - hg))) / hg**2

    sigma = market.vol
    vega = (prix(marche_decale(vol=sigma + pas_vol))
            - prix(marche_decale(vol=sigma - pas_vol))) / (2 * pas_vol)

    r = market.rate
    rho = (prix(marche_decale(rate=r + pas_taux))
           - prix(marche_decale(rate=r - pas_taux))) / (2 * pas_taux)

    # theta = dV/dt = -dV/dT : le temps qui passe réduit la maturité restante.
    theta = -(prix(i=option_decalee(maturity=T + pas_temps))
              - prix(i=option_decalee(maturity=T - pas_temps))) / (2 * pas_temps)

    return Greeks(float(delta), float(gamma), float(vega), float(theta), float(rho))