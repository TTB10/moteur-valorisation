"""Tests de fumée de l'application Streamlit : la page se charge et réagit sans erreur.

Les valeurs numériques sont couvertes par les autres tests ; on vérifie ici l'interface.
Les parties réseau (surface de marché) sont derrière un bouton, donc non exécutées.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parent.parent / "app.py")


def lancer():
    return AppTest.from_file(APP, default_timeout=300).run()


def test_application_europeenne_sans_erreur():
    at = lancer()
    libelles = [m.label for m in at.metric]

    assert not at.exception
    assert "Black-Scholes" in libelles
    assert "Monte Carlo" in libelles
    assert "Arbre CRR" in libelles


def test_application_americaine_sans_erreur():
    at = lancer()
    at.sidebar.radio[1].set_value("américain").run()

    assert not at.exception
    assert "Prime d'exercice anticipé" in [m.label for m in at.metric]


def test_les_six_onglets_sont_presents():
    at = lancer()

    assert not at.exception
    assert len(at.tabs) == 6
