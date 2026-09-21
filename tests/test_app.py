from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parent.parent / "app.py")


def test_application_europeenne_sans_erreur():
    at = AppTest.from_file(APP, default_timeout=120).run()

    assert not at.exception
    assert len(at.metric) == 3                     # Black-Scholes, Monte Carlo, arbre


def test_application_americaine_sans_erreur():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.radio[1].set_value("américain").run()

    assert not at.exception
    assert "Prime d'exercice anticipé" in [m.label for m in at.metric]