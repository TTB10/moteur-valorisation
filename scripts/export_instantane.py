"""Enregistre un instantané de chaîne d'options, pour une démonstration hors ligne."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pricer.market_data_feed import fetch_option_chain

TICKER = "SPY"

brut, spot, horodatage = fetch_option_chain(TICKER)
brut["spot"] = spot
brut["ticker"] = TICKER
brut["capture"] = horodatage.strftime("%Y-%m-%d %H:%M UTC")

dossier = Path("data")
dossier.mkdir(exist_ok=True)
chemin = dossier / "chaine_options_snapshot.csv"
brut.to_csv(chemin, index=False)

print(f"{len(brut)} cotations enregistrées dans {chemin} "
      f"(spot {spot:.2f}, capture {brut['capture'].iloc[0]})")