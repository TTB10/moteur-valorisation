"""Application de démonstration du moteur de valorisation d'options.

Six onglets : valorisation, profils, stratégies, couverture dynamique, volatilité
implicite et risque. L'application ne contient aucune logique financière : elle
appelle le moteur du paquet `pricer`.
"""

import time
from dataclasses import asdict, replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from pricer.engines.analytic import BlackScholesEngine
from pricer.engines.binomial import BinomialTreeEngine
from pricer.engines.monte_carlo import MonteCarloEngine
from pricer.finite_difference import finite_difference_greeks
from pricer.hedging import simulate_delta_hedge
from pricer.implied_vol import NoImpliedVolatility, implied_volatility, price_bounds
from pricer.instruments import AmericanOption, EuropeanOption, Underlying
from pricer.market_data import MarketData
from pricer.plots import (grecques_contre_spot, grecques_contre_temps,
                          payoffs_quatre_positions, prix_contre_spot)
from pricer.portfolio import Portfolio
from pricer.risk import monte_carlo_var, parametric_var
from pricer.strategies import bear_spread, bull_spread, butterfly, straddle, strangle

JOURS_PAR_AN = 252
DEPOT = "https://github.com/TTB10/moteur-valorisation"

st.set_page_config(page_title="Moteur de valorisation d'options", layout="wide",
                   page_icon="📈")

st.title("Moteur de valorisation d'options")
st.markdown(
    f"Le même contrat valorisé par **trois méthodes indépendantes**, puis confronté à ses "
    f"propres hypothèses : couverture discrète, volatilité implicite, mesure du risque. "
    f"[Code, tests et white paper]({DEPOT})"
)

# ------------------------------------------------------------------ paramètres
with st.sidebar:
    st.header("Contrat")
    type_option = st.radio("Type", ["call", "put"], horizontal=True)
    style = st.radio("Exercice", ["européen", "américain"], horizontal=True)
    K = st.number_input("Strike K", min_value=1.0, max_value=1000.0, value=100.0, step=1.0)
    T = st.number_input("Maturité T (années)", min_value=0.02, max_value=10.0, value=1.0, step=0.05)

    st.header("Marché")
    S = st.number_input("Spot S", min_value=1.0, max_value=1000.0, value=100.0, step=1.0)
    r = st.slider("Taux sans risque r (%)", -2.0, 10.0, 5.0, 0.25) / 100
    q = st.slider("Dividende q (%)", 0.0, 10.0, 0.0, 0.25) / 100
    sigma = st.slider("Volatilité σ (%)", 1.0, 100.0, 20.0, 0.5) / 100

    st.header("Méthodes numériques")
    n_paths = st.select_slider("Tirages Monte Carlo", [10_000, 50_000, 100_000, 200_000],
                               value=100_000)
    n_steps = st.select_slider("Pas de l'arbre", [50, 100, 250, 500, 1000], value=500)

    st.divider()
    st.caption(f"[Dépôt GitHub]({DEPOT}) · 1 068 tests automatiques")

americain = style == "américain"
PARAMS = (type_option, americain, K, T, S, r, q, sigma)


def construire(type_option, americain, K, T, S, r, q, sigma):
    """Reconstruit les objets du moteur à partir de paramètres simples (compatibles cache)."""
    marche = MarketData(spot=S, rate=r, dividend=q, vol=sigma)
    classe = AmericanOption if americain else EuropeanOption
    return classe(K, T, type_option), EuropeanOption(K, T, type_option), marche


def chronometre(fonction):
    debut = time.perf_counter()
    resultat = fonction()
    return resultat, (time.perf_counter() - debut) * 1000


def figure(largeur=6.0, hauteur=4.0):
    fig, ax = plt.subplots(figsize=(largeur, hauteur))
    ax.grid(alpha=0.3)
    return fig, ax


# ------------------------------------------------------------------ calculs en cache
@st.cache_data(show_spinner=False)
def valoriser(type_option, americain, K, T, S, r, q, sigma, n_paths, n_steps):
    option, europeenne, marche = construire(type_option, americain, K, T, S, r, q, sigma)
    arbre = BinomialTreeEngine(n_steps)

    if americain:
        prix_am, t_am = chronometre(lambda: arbre.price(option, marche))
        prix_eu = arbre.price(europeenne, marche)
        prix_bs = BlackScholesEngine().price(europeenne, marche)
        return [
            ("Arbre CRR (américaine)", prix_am, f"{t_am:.0f} ms"),
            ("Black-Scholes (européenne)", prix_bs, "formule fermée"),
            ("Prime d'exercice anticipé", prix_am - prix_eu,
             f"{(prix_am - prix_eu) / prix_am:.1%} de la valeur" if prix_am else "—"),
        ]

    prix_bs, t_bs = chronometre(lambda: BlackScholesEngine().price(option, marche))
    moteur_mc = MonteCarloEngine(n_paths, seed=42, antithetic=True, control_variate=True)
    mc, t_mc = chronometre(lambda: moteur_mc.simulate(option, marche))
    prix_arbre, t_arbre = chronometre(lambda: arbre.price(option, marche))
    return [
        ("Black-Scholes", prix_bs, f"exact — {t_bs:.2f} ms"),
        ("Monte Carlo", mc.price, f"± {1.96 * mc.std_error:.4f} (IC 95 %) — {t_mc:.0f} ms"),
        ("Arbre CRR", prix_arbre, f"écart {prix_arbre - prix_bs:+.4f} — {t_arbre:.0f} ms"),
    ]


@st.cache_data(show_spinner=False)
def table_grecques(type_option, americain, K, T, S, r, q, sigma, n_steps):
    option, _, marche = construire(type_option, americain, K, T, S, r, q, sigma)
    # Sur un moteur discret, le gamma exige un pas plus grand : le bruit joue le rôle
    # d'un epsilon effectif très supérieur à la précision machine.
    fd = finite_difference_greeks(BinomialTreeEngine(n_steps), option, marche,
                                  pas_spot=1e-3, pas_spot_gamma=1e-2)
    colonnes = {}
    if not americain:
        colonnes["Analytique (Black-Scholes)"] = asdict(BlackScholesEngine().greeks(option, marche))
    colonnes["Différences finies (arbre)"] = asdict(fd)
    df = pd.DataFrame(colonnes)
    df.index = ["Delta", "Gamma", "Vega", "Thêta", "Rhô"]
    return df


@st.cache_data(show_spinner=False)
def convergence_mc(type_option, K, T, S, r, q, sigma):
    _, option, marche = construire(type_option, False, K, T, S, r, q, sigma)
    tailles = np.unique(np.logspace(2, 5.3, 12).astype(int) // 2 * 2)
    simple = [MonteCarloEngine(int(n), seed=7).simulate(option, marche).std_error for n in tailles]
    combine = [MonteCarloEngine(int(n), seed=7, antithetic=True, control_variate=True)
               .simulate(option, marche).std_error for n in tailles]
    return tailles, np.array(simple), np.array(combine)


@st.cache_data(show_spinner=False)
def convergence_arbre(type_option, americain, K, T, S, r, q, sigma):
    option, europeenne, marche = construire(type_option, americain, K, T, S, r, q, sigma)
    pas = np.arange(10, 301)
    prix = np.array([BinomialTreeEngine(int(n)).price(option, marche) for n in pas])
    if americain:
        return pas, prix, BinomialTreeEngine(2000).price(option, marche), "Arbre à 2000 pas"
    return pas, prix, BlackScholesEngine().price(europeenne, marche), "Black-Scholes"


def figure_profil(vue, K, T, S, r, q, sigma):
    marche = MarketData(spot=S, rate=r, dividend=q, vol=sigma)
    bornes = np.linspace(max(K * 0.4, 1.0), K * 1.6, 300)

    if vue == "Grecques contre le spot":
        return grecques_contre_spot(marche, strike=K, maturite=T, spots=bornes)
    if vue == "Grecques contre le temps":
        return grecques_contre_temps(marche, strike=K, moneyness=(0.9 * K, K, 1.1 * K))
    if vue == "Prix contre le spot":
        return prix_contre_spot(marche, strike=K, spots=bornes)
    return payoffs_quatre_positions(strike=K, spots=bornes)


STRATEGIES = {
    "Bull spread": lambda K, T: bull_spread(0.95 * K, 1.05 * K, T),
    "Bear spread": lambda K, T: bear_spread(0.95 * K, 1.05 * K, T),
    "Straddle": lambda K, T: straddle(K, T),
    "Strangle": lambda K, T: strangle(0.90 * K, 1.10 * K, T),
    "Butterfly": lambda K, T: butterfly(0.90 * K, K, 1.10 * K, T),
    "Call seul": lambda K, T: Portfolio.from_pairs(
        [(EuropeanOption(K, T, "call"), 1)], name="Call acheté"),
}


@st.cache_data(show_spinner=False)
def analyser_strategie(nom, K, T, S, r, q, sigma):
    marche = MarketData(spot=S, rate=r, dividend=q, vol=sigma)
    pf = STRATEGIES[nom](K, T)
    spots = np.linspace(max(K * 0.5, 1.0), K * 1.5, 300)
    grecques = asdict(pf.greeks(marche))
    profil = pd.DataFrame({
        "spot": spots,
        "payoff": pf.payoff(spots),
        "pnl": pf.payoff(spots) - pf.price(marche) * np.exp(r * T),
    })
    courbes = {nom_g: np.array([getattr(pf.greeks(replace(marche, spot=s)), nom_g) for s in spots])
               for nom_g in ("delta", "gamma", "vega", "theta")}
    return pf.name, pf.price(marche), grecques, profil, courbes


@st.cache_data(show_spinner=False)
def experience_couverture(K, T, S, r, q, sigma, frequences, n_paths_hedge):
    option = EuropeanOption(K, T, "call")
    marche = MarketData(spot=S, rate=r, dividend=q, vol=sigma)
    resultats = {}
    for n in frequences:
        res = simulate_delta_hedge(option, marche, int(n), n_paths=int(n_paths_hedge), seed=2026)
        resultats[int(n)] = {
            "errors": res.errors, "mean": res.mean, "std": res.std,
            "q01": res.quantile(0.01), "q99": res.quantile(0.99), "premium": res.premium,
        }
    return resultats


@st.cache_data(show_spinner=False)
def pnl_contre_vol_realisee(K, T, S, r, q, sigma, n_rebal, n_paths_hedge):
    option = EuropeanOption(K, T, "call")
    vols = np.linspace(max(sigma - 0.10, 0.02), sigma + 0.10, 9)
    moyennes, ecarts = [], []
    for vol_reelle in vols:
        marche_reel = MarketData(spot=S, rate=r, dividend=q, vol=float(vol_reelle))
        res = simulate_delta_hedge(option, marche_reel, int(n_rebal),
                                   n_paths=int(n_paths_hedge), seed=2026, hedge_vol=sigma)
        moyennes.append(res.mean)
        ecarts.append(res.std)
    return vols, np.array(moyennes), np.array(ecarts)


PORTEFEUILLES_RISQUE = {
    "Call acheté": lambda K, T, m: Portfolio.from_pairs(
        [(EuropeanOption(K, T, "call"), 1)], name="Call acheté"),
    "Straddle vendu, couvert en delta": lambda K, T, m: Portfolio.from_pairs(
        [(straddle(K, T), -1)], name="Straddle vendu").delta_hedged(m),
    "Action seule": lambda K, T, m: Portfolio.from_pairs(
        [(Underlying(), 1)], name="Action"),
    "Butterfly acheté": lambda K, T, m: butterfly(0.9 * K, K, 1.1 * K, T),
}


@st.cache_data(show_spinner=False)
def mesurer_risque(nom_pf, K, T, S, r, q, sigma, confiance, n_scenarios):
    marche = MarketData(spot=S, rate=r, dividend=q, vol=sigma)
    pf = PORTEFEUILLES_RISQUE[nom_pf](K, T, marche)
    vol_jour = sigma / np.sqrt(JOURS_PAR_AN)
    param = parametric_var(pf, marche, vol_jour, confidence=confiance)
    mc = monte_carlo_var(pf, marche, vol_jour, confidence=confiance,
                         n_scenarios=int(n_scenarios), seed=2026)
    return {
        "valeur": pf.price(marche),
        "delta": pf.greeks(marche).delta,
        "gamma": pf.greeks(marche).gamma,
        "param": (param.var, param.es),
        "mc": (mc.var, mc.es),
        "pertes": mc.losses,
    }


# ------------------------------------------------------------------ onglets
onglets = st.tabs(["Valorisation", "Profils", "Stratégies", "Couverture",
                   "Volatilité implicite", "Risque"])

# ------------------------------------------------------- 1. Valorisation
with onglets[0]:
    try:
        resultats = valoriser(*PARAMS, n_paths, n_steps)
    except ValueError as erreur:
        st.error(f"Paramètres incohérents : {erreur}")
        st.stop()

    if americain:
        st.info("Black-Scholes et Monte Carlo ne gèrent pas l'exercice anticipé : "
                "seul l'arbre valorise l'option américaine. L'écart avec l'européenne "
                "est la prime d'exercice anticipé.")
    for colonne, (nom, prix, detail) in zip(st.columns(len(resultats)), resultats):
        colonne.metric(nom, f"{prix:.4f}")
        colonne.caption(detail)

    st.subheader("Grecques")
    st.dataframe(table_grecques(*PARAMS, n_steps).style.format("{:.4f}"),
                 use_container_width=True)
    st.caption("Conventions : vega par 1,00 de volatilité (÷100 par point), thêta par an "
               "(÷365 par jour), rhô par 1,00 de taux. Les différences finies fonctionnent "
               "avec n'importe quel moteur, y compris pour une américaine, pour laquelle "
               "aucune formule analytique n'existe.")

    st.subheader("Convergence")
    gauche, droite = st.columns(2)
    with gauche:
        if americain:
            st.info("Monte Carlo n'est pas défini pour l'exercice anticipé.")
        else:
            tailles, simple, combine = convergence_mc(type_option, K, T, S, r, q, sigma)
            fig, ax = figure()
            ax.loglog(tailles, simple, "o-", markersize=3, label="Monte Carlo simple")
            ax.loglog(tailles, combine, "o-", markersize=3, label="Antithétiques + contrôle")
            ax.loglog(tailles, simple[-1] * np.sqrt(tailles[-1] / tailles), "k:",
                      label="Pente −1/2")
            ax.set_xlabel("Nombre de tirages")
            ax.set_ylabel("Erreur standard")
            ax.set_title("Monte Carlo : convergence en 1/√n")
            ax.legend(fontsize=8)
            st.pyplot(fig)
            plt.close(fig)
            gain = (simple[-1] / combine[-1]) ** 2
            st.caption(f"Gain de variance de la réduction combinée : **{gain:.0f}×** "
                       "à budget de calcul égal.")
    with droite:
        pas, prix_arbre, reference, libelle = convergence_arbre(*PARAMS)
        fig, ax = figure()
        ax.plot(pas, prix_arbre, linewidth=0.8, label="Arbre CRR")
        ax.axhline(reference, color="black", linestyle="--", linewidth=1, label=libelle)
        ax.set_xlabel("Nombre de pas N")
        ax.set_ylabel("Prix")
        ax.set_title("Arbre : convergence en 1/N, oscillante")
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close(fig)
        st.caption("L'erreur alterne de signe selon la parité de N : moyenner les arbres "
                   "à N et N+1 pas divise l'erreur par environ 15.")

# ------------------------------------------------------- 2. Profils
with onglets[1]:
    st.markdown("Payoffs, prix et grecques tracés avec les paramètres de la barre latérale.")
    choix = st.radio("Vue", ["Grecques contre le spot", "Grecques contre le temps",
                             "Prix contre le spot", "Payoffs élémentaires"],
                     horizontal=True)

    with st.spinner("Tracé en cours…"):
        fig = figure_profil(choix, K, T, S, r, q, sigma)
    st.pyplot(fig)
    plt.close(fig)

    LEGENDES = {
        "Grecques contre le spot":
            "Gamma et vega sont identiques pour le call et le put, et en cloche autour de la "
            "monnaie : maximum exact en d₁ = −σ√T pour le gamma, +σ√T pour le vega.",
        "Grecques contre le temps":
            "Gamma et thêta d'une option à la monnaie divergent quand la maturité tend vers "
            "zéro, alors qu'ils s'annulent hors de la monnaie : couvrir une option ATM en fin "
            "de vie est le cas le plus délicat.",
        "Prix contre le spot":
            "L'écart entre le prix et la valeur intrinsèque est la valeur temps : elle rémunère "
            "l'incertitude restante et s'annule à maturité.",
        "Payoffs élémentaires":
            "L'asymétrie acheteur / vendeur : perte bornée à la prime contre gain borné à la "
            "prime, avec un risque potentiellement illimité côté vendeur.",
    }
    st.caption(LEGENDES[choix])

# ------------------------------------------------------- 3. Stratégies
with onglets[2]:
    nom_strategie = st.selectbox("Stratégie", list(STRATEGIES), index=2)
    nom, prime, grecques, profil, courbes = analyser_strategie(
        nom_strategie, K, T, S, r, q, sigma)

    cols = st.columns(5)
    cols[0].metric("Prime", f"{prime:.2f}")
    for col, (cle, libelle) in zip(cols[1:], [("delta", "Delta"), ("gamma", "Gamma"),
                                              ("vega", "Vega"), ("theta", "Thêta")]):
        col.metric(libelle, f"{grecques[cle]:+.4f}" if cle == "gamma"
                   else f"{grecques[cle]:+.3f}")

    gauche, droite = st.columns(2)
    with gauche:
        fig, ax = figure()
        ax.plot(profil["spot"], profil["payoff"], linewidth=1.8, label="Payoff à maturité")
        ax.plot(profil["spot"], profil["pnl"], "--", linewidth=1.2,
                label="P&L net de la prime capitalisée")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(S, color="grey", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Spot à maturité")
        ax.set_ylabel("Gain")
        ax.set_title(nom)
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close(fig)
    with droite:
        grecque_tracee = st.selectbox("Grecque", ["gamma", "delta", "vega", "theta"])
        fig, ax = figure()
        ax.plot(profil["spot"], courbes[grecque_tracee], linewidth=1.5)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(S, color="grey", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Spot")
        ax.set_ylabel(grecque_tracee.capitalize())
        ax.set_title(f"{grecque_tracee.capitalize()} agrégé contre le spot")
        st.pyplot(fig)
        plt.close(fig)

    st.caption("Les grecques d'un portefeuille sont la somme pondérée de celles de ses lignes, "
               "par linéarité de la dérivation. Le butterfly est la seule de ces stratégies à "
               "thêta positif : net vendeur d'options à la monnaie, il encaisse la valeur temps "
               "et paie le risque de mouvement.")

# ------------------------------------------------------- 4. Couverture
with onglets[3]:
    st.markdown("**Expérience.** On vend un call, on encaisse la prime et on la réplique en "
                "rebalançant le delta à fréquence fixe. L'erreur finale est ce qui reste après "
                "paiement du payoff.")
    reglages = st.columns(2)
    n_paths_hedge = reglages[0].select_slider("Trajectoires simulées",
                                              [2_000, 5_000, 10_000], value=5_000)
    # Plafond volontaire : la simulation alloue une matrice n_paths x N (mémoire limitée en ligne).
    montrer_vol = reglages[1].checkbox("Étudier l'écart entre volatilité de couverture et "
                                       "volatilité réalisée", value=False)

    frequences = (12, 52, 252, 1008)
    with st.spinner("Simulation des trajectoires…"):
        resultats_hedge = experience_couverture(K, T, S, r, q, sigma, frequences, n_paths_hedge)

    prime_hedge = resultats_hedge[frequences[0]]["premium"]
    st.caption(f"Prime encaissée : **{prime_hedge:.4f}**")

    tableau = pd.DataFrame([
        {"Fréquence": libelle, "Moyenne": v["mean"], "Écart-type": v["std"],
         "En % de la prime": v["std"] / prime_hedge, "Quantile 1 %": v["q01"],
         "Quantile 99 %": v["q99"]}
        for libelle, v in zip(["Mensuelle (12)", "Hebdomadaire (52)", "Quotidienne (252)",
                               "4 fois par jour (1008)"], resultats_hedge.values())
    ]).set_index("Fréquence")
    st.dataframe(tableau.style.format({"Moyenne": "{:+.4f}", "Écart-type": "{:.4f}",
                                       "En % de la prime": "{:.1%}",
                                       "Quantile 1 %": "{:.3f}", "Quantile 99 %": "{:.3f}"}),
                 use_container_width=True)

    gauche, droite = st.columns(2)
    with gauche:
        fig, ax = figure()
        for (n, v), libelle in zip(resultats_hedge.items(),
                                   ["Mensuel", "Hebdomadaire", "Quotidien", "4×/jour"]):
            ax.hist(v["errors"], bins=120, range=(-3 * prime_hedge / 10, 3 * prime_hedge / 10),
                    density=True, histtype="step", linewidth=1.4,
                    label=f"{libelle} : σ = {v['std']:.3f}")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Erreur de couverture")
        ax.set_ylabel("Densité")
        ax.set_title("Distribution selon la fréquence")
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close(fig)
    with droite:
        ns = np.array(frequences, dtype=float)
        stds = np.array([v["std"] for v in resultats_hedge.values()])
        fig, ax = figure()
        ax.loglog(ns, stds, "o-", label="Écart-type mesuré")
        ax.loglog(ns, stds[-1] * np.sqrt(ns[-1] / ns), "k:", label="Pente −1/2")
        ax.set_xlabel("Nombre de rebalancements N")
        ax.set_ylabel("Écart-type de l'erreur")
        ax.set_title("L'erreur décroît en 1/√N : elle ne disparaît jamais")
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close(fig)

    if montrer_vol:
        with st.spinner("Variation de la volatilité réalisée…"):
            vols, moyennes, ecarts = pnl_contre_vol_realisee(K, T, S, r, q, sigma, 252,
                                                             n_paths_hedge)
        fig, ax = figure(12, 4)
        ax.plot(vols * 100, moyennes, "o-", label="P&L moyen du vendeur")
        ax.fill_between(vols * 100, moyennes - ecarts, moyennes + ecarts, alpha=0.2,
                        label="± 1 écart-type")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(sigma * 100, color="grey", linestyle=":",
                   label=f"Volatilité de couverture ({sigma:.0%})")
        ax.set_xlabel("Volatilité réalisée (%)")
        ax.set_ylabel("P&L du vendeur")
        ax.set_title("Un vendeur couvert parie sur l'agitation, pas sur la direction")
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Le P&L suit environ ½∫ΓS²(σ²_implicite − σ²_réalisée) dt : il s'annule "
                   "quand la volatilité réalisée égale celle à laquelle l'option a été vendue.")

    st.caption("Le vendeur est short gamma : la couverture se refait toujours à son "
               "désavantage, ce que le thêta compense en moyenne seulement.")

# ------------------------------------------------------- 5. Volatilité implicite
with onglets[4]:
    st.markdown("**Inversion de Black-Scholes.** On cherche la volatilité qui reproduit un prix "
                "observé, par Newton-Raphson maintenu dans un encadrement (repli dichotomique).")

    option_iv = EuropeanOption(K, T, type_option)
    marche_iv = MarketData(spot=S, rate=r, dividend=q, vol=sigma)
    borne_basse, borne_haute = price_bounds(option_iv, marche_iv)
    prix_theorique = BlackScholesEngine().price(option_iv, marche_iv)

    st.caption(f"Bornes d'arbitrage pour ce contrat : **{borne_basse:.4f}** à "
               f"**{borne_haute:.4f}**. Au-delà, aucune volatilité ne peut reproduire le prix.")
    prix_observe = st.number_input(
        "Prix observé", min_value=0.0, max_value=float(borne_haute * 1.2),
        value=float(round(prix_theorique, 4)), step=0.01, format="%.4f")

    try:
        iv = implied_volatility(prix_observe, option_iv, marche_iv)
        vega = BlackScholesEngine().greeks(option_iv, replace(marche_iv, vol=iv)).vega
        cols = st.columns(3)
        cols[0].metric("Volatilité implicite", f"{iv:.2%}")
        cols[1].metric("Vega au point trouvé", f"{vega:.2f}")
        precision = 0.01 / vega if vega > 0 else float("inf")
        cols[2].metric("Précision pour 1 cent d'écart", f"{precision:.2%}")
        st.caption("La précision atteignable sur la volatilité implicite vaut environ "
                   "Δprix / vega. C'est pourquoi les surfaces de marché se construisent sur les "
                   "options **hors de la monnaie** : elles portent le vega, et par parité elles "
                   "contiennent la même information que les options dans la monnaie.")
    except NoImpliedVolatility as erreur:
        st.warning(f"Pas de volatilité implicite exploitable : {erreur}")

    st.divider()
    st.markdown("**Surface de marché.** Les prix cotés sont récupérés en direct, nettoyés par "
                "sept filtres journalisés, puis inversés. Le forward de chaque maturité est "
                "estimé par parité call-put, sans supposer de dividende.")
    ticker = st.text_input("Sous-jacent", value="SPY")
    taux_surface = st.slider("Taux sans risque supposé (%)", 0.0, 8.0, 4.0, 0.25) / 100

    if st.button("Charger la surface de volatilité"):
        try:
            with st.spinner(f"Récupération des chaînes d'options de {ticker}…"):
                from pricer.market_data_feed import clean_option_chain, fetch_option_chain
                from pricer.vol_surface import (build_vol_surface,
                                                butterfly_arbitrage_violations,
                                                calendar_arbitrage_violations)

                brut, spot_marche, horodatage = fetch_option_chain(ticker)
                propre, journal = clean_option_chain(brut, spot_marche)
                avec_itm, _ = clean_option_chain(brut, spot_marche, garder_otm_seulement=False)
                surface, journal_surface = build_vol_surface(
                    propre, spot_marche, taux_surface, propre_pour_parite=avec_itm)

            st.success(f"{ticker} — spot {spot_marche:.2f} — "
                       f"capture {horodatage:%Y-%m-%d %H:%M UTC}")

            cols = st.columns(3)
            cols[0].metric("Cotations brutes", journal["total initial"])
            cols[1].metric("Points retenus", journal_surface["points retenus"])
            cols[2].metric("Maturités", len(surface.maturities))

            repli = journal_surface.get("forwards estimés par repli (sans parité)", 0)
            if repli or journal_surface["points retenus"] < 50:
                st.warning(
                    f"Données trop pauvres pour une surface fiable : "
                    f"{journal_surface['points retenus']} points retenus, "
                    f"{repli} maturité(s) sans forward estimable par parité "
                    "(le forward est alors supposé sans dividende, ce qui fausse la moneyness). "
                    "Cause la plus probable : marché fermé, carnets vides ou cotations périmées. "
                    "Les options américaines cotent de 15 h 30 à 22 h, heure de Paris."
                )

            T_obs = np.array(surface.maturities)
            lnF = np.array([np.log(surface.forwards[t]) for t in T_obs])
            pente, ordonnee = np.polyfit(T_obs, lnF, 1)
            st.caption(f"Régression des forwards : taux de portage **r − q = {pente:.2%}**, "
                       f"spot implicite **{np.exp(ordonnee):.2f}** contre {spot_marche:.2f} coté. "
                       "Le marché ne révèle que la différence r − q, jamais les deux séparément.")

            gauche, droite = st.columns(2)
            with gauche:
                fig, ax = figure(6, 4.5)
                couleurs = plt.cm.viridis(np.linspace(0, 0.9, len(surface.maturities)))
                for maturite, couleur in zip(surface.maturities, couleurs):
                    tranche = surface.smile(maturite)
                    ax.plot(tranche["log_moneyness"], tranche["iv"] * 100, "o-", markersize=3,
                            color=couleur, linewidth=1.2, label=f"T = {maturite:.2f}")
                ax.axvline(0, color="grey", linestyle=":", linewidth=1)
                ax.set_xlabel("Log-moneyness ln(K / F)")
                ax.set_ylabel("Volatilité implicite (%)")
                ax.set_title("Smile par maturité")
                ax.legend(fontsize=7)
                st.pyplot(fig)
                plt.close(fig)
            with droite:
                atm = surface.atm_term_structure()
                fig, ax = figure(6, 4.5)
                ax.plot(atm["maturity"], atm["iv_atm"] * 100, "o-", linewidth=1.5)
                ax.set_xlabel("Maturité (années)")
                ax.set_ylabel("Volatilité implicite à la monnaie (%)")
                ax.set_title("Structure par terme")
                st.pyplot(fig)
                plt.close(fig)

            papillon = sum(butterfly_arbitrage_violations(surface, t)
                           for t in surface.maturities)
            st.caption(f"Tests d'absence d'arbitrage : **{papillon}** violation(s) de convexité "
                       f"en strike (nettes des coûts d'exécution), "
                       f"**{calendar_arbitrage_violations(surface)}** violation(s) calendaire(s). "
                       "Si Black-Scholes était exact, cette surface serait plate.")
            with st.expander("Journal de nettoyage"):
                st.dataframe(pd.DataFrame({"Cotations": {**journal, **journal_surface}}),
                             use_container_width=True)
        except Exception as erreur:  # réseau indisponible, ticker inconnu, marché fermé
            st.error(f"Chargement impossible : {erreur}")

# ------------------------------------------------------- 6. Risque
with onglets[5]:
    st.markdown("**VaR et Expected Shortfall à un jour**, par réévaluation complète du "
                "portefeuille (le spot est choqué, les options vieillies d'un jour).")
    reglages = st.columns(3)
    nom_pf = reglages[0].selectbox("Portefeuille", list(PORTEFEUILLES_RISQUE), index=1)
    confiance = reglages[1].select_slider("Niveau de confiance", [0.95, 0.975, 0.99],
                                          value=0.99)
    n_scenarios = reglages[2].select_slider("Scénarios simulés", [2_000, 5_000, 10_000],
                                            value=5_000)

    with st.spinner("Réévaluation des scénarios…"):
        risque = mesurer_risque(nom_pf, K, T, S, r, q, sigma, confiance, n_scenarios)

    cols = st.columns(4)
    cols[0].metric("Valeur du portefeuille", f"{risque['valeur']:.4f}")
    cols[1].metric("Delta / Gamma",
                   f"{risque['delta']:+.3f} / {risque['gamma']:+.4f}")
    cols[2].metric(f"VaR {confiance:.1%} paramétrique", f"{risque['param'][0]:.4f}")
    cols[3].metric(f"VaR {confiance:.1%} Monte Carlo", f"{risque['mc'][0]:.4f}")

    comparaison = pd.DataFrame({
        "Paramétrique (delta-normale)": {"VaR": risque["param"][0], "ES": risque["param"][1]},
        "Monte Carlo (réévaluation complète)": {"VaR": risque["mc"][0], "ES": risque["mc"][1]},
    })
    st.dataframe(comparaison.style.format("{:.4f}"), use_container_width=True)

    fig, ax = figure(12, 4)
    ax.hist(risque["pertes"], bins=120, density=True, color="steelblue", alpha=0.8)
    ax.axvline(risque["mc"][0], color="darkorange", linewidth=1.5,
               label=f"VaR {confiance:.1%} = {risque['mc'][0]:.4f}")
    ax.axvline(risque["mc"][1], color="firebrick", linewidth=1.5,
               label=f"ES {confiance:.1%} = {risque['mc'][1]:.4f}")
    ax.set_xlabel("Perte sur un jour")
    ax.set_ylabel("Densité")
    ax.set_title("Distribution simulée des pertes")
    ax.legend(fontsize=8)
    st.pyplot(fig)
    plt.close(fig)

    if abs(risque["delta"]) < 1e-9 and risque["mc"][0] > 0:
        st.warning("Le portefeuille est delta-neutre : la méthode paramétrique conclut à un "
                   "risque nul, alors que la réévaluation complète mesure une perte réelle. "
                   "C'est la convexité (le gamma) que la méthode delta-normale ignore.")
    st.caption("La VaR n'est pas sous-additive : deux pertes indépendantes de probabilité 4 % "
               "ont chacune une VaR à 95 % nulle, mais leur somme une VaR strictement positive. "
               "L'Expected Shortfall, elle, est cohérente, ce qui a conduit la réglementation "
               "bâloise à l'adopter au seuil de 97,5 %.")

st.divider()
st.caption("Limites du modèle : volatilité constante, absence de sauts, couverture continue, "
           "coûts de transaction ignorés. Le white paper du dépôt les met à l'épreuve sur "
           "données réelles.")
