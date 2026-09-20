"""Récupération et nettoyage des chaînes d'options cotées.

Séparation volontaire :
  fetch_option_chain  -> dépend du réseau, non testable hors ligne
  clean_option_chain  -> pure transformation d'un DataFrame, entièrement testable
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

COLONNES = ["type", "strike", "maturity", "bid", "ask", "mid", "volume", "open_interest"]


def fetch_option_chain(ticker, max_maturities=8):
    """Télécharge la chaîne d'options et le spot. Renvoie (DataFrame brut, spot, horodatage)."""
    import yfinance as yf

    actif = yf.Ticker(ticker)
    spot = _spot_courant(actif)
    echeances = actif.options[:max_maturities]
    if not echeances:
        raise ValueError(f"Aucune échéance cotée pour {ticker}.")

    aujourdhui = datetime.now(timezone.utc).date()
    lignes = []

    for echeance in echeances:
        chaine = actif.option_chain(echeance)
        maturite = (pd.Timestamp(echeance).date() - aujourdhui).days / 365.0
        if maturite <= 0:
            continue
        for type_option, table in (("call", chaine.calls), ("put", chaine.puts)):
            extrait = table[["strike", "bid", "ask", "volume", "openInterest"]].copy()
            extrait["type"] = type_option
            extrait["maturity"] = maturite
            extrait["expiration"] = echeance
            lignes.append(extrait)

    brut = pd.concat(lignes, ignore_index=True)
    brut = brut.rename(columns={"openInterest": "open_interest"})
    return brut, spot, datetime.now(timezone.utc)


def _spot_courant(actif):
    """Dernier prix connu du sous-jacent."""
    historique = actif.history(period="1d")
    if historique.empty:
        raise ValueError("Impossible de récupérer le spot.")
    return float(historique["Close"].iloc[-1])


def clean_option_chain(brut, spot, volume_min=10, spread_max_relatif=0.25,
                       moneyness_max=0.5, garder_otm_seulement=True):
    """Filtre les cotations inexploitables. Renvoie (DataFrame propre, journal des rejets).

    Critères, dans l'ordre :
      1. cotation absente ou incohérente (bid <= 0, ask <= bid)
      2. illiquidité (volume insuffisant)
      3. écart achat-vente excessif, relatif au mid
      4. strike trop éloigné du spot (|ln(K/S)| > moneyness_max)
      5. option dans la monnaie : vega faible, information redondante par parité
    """
    df = brut.copy()
    journal = {"total initial": len(df)}

    def rejeter(masque, motif):
        nonlocal df
        avant = len(df)
        df = df[masque].copy()
        journal[motif] = avant - len(df)

    rejeter((df["bid"] > 0) & (df["ask"] > df["bid"]), "cotation absente ou incohérente")

    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    df["spread_relatif"] = (df["ask"] - df["bid"]) / df["mid"]

    rejeter(df["volume"].fillna(0) >= volume_min, "volume insuffisant")
    rejeter(df["spread_relatif"] <= spread_max_relatif, "écart achat-vente excessif")

    df["log_moneyness"] = np.log(df["strike"] / spot)
    rejeter(df["log_moneyness"].abs() <= moneyness_max, "strike trop éloigné")

    if garder_otm_seulement:
        est_otm = ((df["type"] == "call") & (df["strike"] >= spot)) | \
                  ((df["type"] == "put") & (df["strike"] < spot))
        rejeter(est_otm, "option dans la monnaie (vega faible)")

    journal["total retenu"] = len(df)
    colonnes = [c for c in COLONNES if c in df.columns] + \
               ["spread_relatif", "log_moneyness", "expiration"]
    return df[[c for c in colonnes if c in df.columns]].reset_index(drop=True), journal


def implied_forward_by_parity(df_propre, maturite, rate):
    """Estime le forward implicite par parité call-put, plutôt que de supposer le dividende.

    Pour un strike coté à la fois en call et en put : F = K + e^{rT} (C - P).
    On prend la médiane sur les strikes disponibles, robuste aux valeurs aberrantes.
    """
    tranche = df_propre[np.isclose(df_propre["maturity"], maturite)]
    calls = tranche[tranche["type"] == "call"].set_index("strike")["mid"]
    puts = tranche[tranche["type"] == "put"].set_index("strike")["mid"]
    communs = calls.index.intersection(puts.index)

    if len(communs) == 0:
        return None

    forwards = communs.values + np.exp(rate * maturite) * (calls[communs] - puts[communs]).values
    return float(np.median(forwards))