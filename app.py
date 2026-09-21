"""Application de démonstration : prix, grecques et convergence des trois moteurs."""

import time
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from pricer.engines.analytic import BlackScholesEngine
from pricer.engines.binomial import BinomialTreeEngine
from pricer.engines.monte_carlo import MonteCarloEngine
from pricer.finite_difference import finite_difference_greeks
from pricer.instruments import AmericanOption, EuropeanOption
from pricer.market_data import MarketData

st.set_page_config(page_title="Moteur de valorisation d'options", layout="wide")
st.title("Moteur de valorisation d'options")
st.caption("Le même contrat valorisé par trois méthodes indépendantes. "
           "Code, tests et white paper : github.com/TTB10/moteur-valorisation")

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

americain = style == "américain"


def construire(type_option, americain, K, T, S, r, q, sigma):
    marche = MarketData(spot=S, rate=r, dividend=q, vol=sigma)
    classe = AmericanOption if americain else EuropeanOption
    return classe(K, T, type_option), EuropeanOption(K, T, type_option), marche


def chronometre(fonction):
    debut = time.perf_counter()
    resultat = fonction()
    return resultat, (time.perf_counter() - debut) * 1000


# ------------------------------------------------------------------ calculs (mis en cache)
@st.cache_data
def valoriser(type_option, americain, K, T, S, r, q, sigma, n_paths, n_steps):
    option, europeenne, marche = construire(type_option, americain, K, T, S, r, q, sigma)
    arbre = BinomialTreeEngine(n_steps)

    if americain:
        prix_am, t_am = chronometre(lambda: arbre.price(option, marche))
        prix_eu, t_eu = chronometre(lambda: arbre.price(europeenne, marche))
        prix_bs = BlackScholesEngine().price(europeenne, marche)
        return [
            ("Arbre CRR (américaine)", prix_am, f"{t_am:.1f} ms"),
            ("Black-Scholes (européenne)", prix_bs, "formule fermée"),
            ("Prime d'exercice anticipé", prix_am - prix_eu,
             f"{(prix_am - prix_eu) / prix_am:.1%} de la valeur"),
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


@st.cache_data
def grecques(type_option, americain, K, T, S, r, q, sigma, n_steps):
    option, _, marche = construire(type_option, americain, K, T, S, r, q, sigma)
    # Sur l'arbre (moteur discret), le gamma demande un pas plus grand : voir le white paper.
    fd = finite_difference_greeks(BinomialTreeEngine(n_steps), option, marche,
                                  pas_spot=1e-3, pas_spot_gamma=1e-2)
    table = {}
    if not americain:
        table["Analytique (Black-Scholes)"] = asdict(BlackScholesEngine().greeks(option, marche))
    table["Différences finies (arbre)"] = asdict(fd)
    df = pd.DataFrame(table)
    df.index = ["Delta", "Gamma", "Vega", "Thêta", "Rhô"]
    return df


@st.cache_data
def convergence_mc(type_option, K, T, S, r, q, sigma):
    _, option, marche = construire(type_option, False, K, T, S, r, q, sigma)
    tailles = np.unique(np.logspace(2, 5.3, 12).astype(int) // 2 * 2)
    simple = [MonteCarloEngine(int(n), seed=7).simulate(option, marche).std_error for n in tailles]
    combine = [MonteCarloEngine(int(n), seed=7, antithetic=True, control_variate=True)
               .simulate(option, marche).std_error for n in tailles]
    return tailles, np.array(simple), np.array(combine)


@st.cache_data
def convergence_arbre(type_option, americain, K, T, S, r, q, sigma):
    option, europeenne, marche = construire(type_option, americain, K, T, S, r, q, sigma)
    pas = np.arange(10, 301)
    prix = np.array([BinomialTreeEngine(int(n)).price(option, marche) for n in pas])
    if americain:
        reference = BinomialTreeEngine(2000).price(option, marche)
        libelle = "Arbre à 2000 pas"
    else:
        reference = BlackScholesEngine().price(europeenne, marche)
        libelle = "Black-Scholes"
    return pas, prix, reference, libelle


# ------------------------------------------------------------------ affichage
parametres = (type_option, americain, K, T, S, r, q, sigma)

try:
    resultats = valoriser(*parametres, n_paths, n_steps)
except ValueError as erreur:
    st.error(f"Paramètres incohérents : {erreur}")
    st.stop()

st.subheader("Prix")
if americain:
    st.info("Black-Scholes et Monte Carlo ne gèrent pas l'exercice anticipé : "
            "seul l'arbre valorise l'option américaine.")
for colonne, (nom, prix, detail) in zip(st.columns(len(resultats)), resultats):
    colonne.metric(nom, f"{prix:.4f}")
    colonne.caption(detail)

st.subheader("Grecques")
st.dataframe(grecques(*parametres, n_steps).style.format("{:.4f}"), use_container_width=True)
st.caption("Conventions : vega par 1,00 de volatilité (÷100 par point), thêta par an "
           "(÷365 par jour), rhô par 1,00 de taux. Les différences finies sur l'arbre "
           "fonctionnent aussi pour l'américaine, pour laquelle aucune formule n'existe.")

st.subheader("Convergence")
gauche, droite = st.columns(2)

with gauche:
    if americain:
        st.info("Monte Carlo n'est pas défini pour l'exercice anticipé.")
    else:
        tailles, simple, combine = convergence_mc(type_option, K, T, S, r, q, sigma)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.loglog(tailles, simple, "o-", markersize=3, label="Monte Carlo simple")
        ax.loglog(tailles, combine, "o-", markersize=3, label="Antithétiques + contrôle")
        ax.loglog(tailles, simple[-1] * np.sqrt(tailles[-1] / tailles), "k:", label="Pente −1/2")
        ax.set_xlabel("Nombre de tirages")
        ax.set_ylabel("Erreur standard")
        ax.set_title("Monte Carlo : convergence en 1/√n")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close(fig)

with droite:
    pas, prix, reference, libelle = convergence_arbre(*parametres)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(pas, prix, linewidth=0.8, label="Arbre CRR")
    ax.axhline(reference, color="black", linestyle="--", linewidth=1, label=libelle)
    ax.set_xlabel("Nombre de pas N")
    ax.set_ylabel("Prix")
    ax.set_title("Arbre : convergence en 1/N, oscillante")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    st.pyplot(fig)
    plt.close(fig)

st.divider()
st.caption("Limites du modèle : volatilité constante, absence de sauts, couverture continue, "
           "coûts de transaction ignorés. Le white paper du dépôt les met à l'épreuve sur données réelles.")