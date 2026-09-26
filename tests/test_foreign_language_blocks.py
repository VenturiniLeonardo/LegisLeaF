"""Test del filtro dei blocchi in lingua diversa da quella del documento (f2).

Caso che ha motivato il codice: alcune fonti in inglese pubblicano ogni
disposizione anche in un'altra lingua ufficiale (es. le leggi federali
canadesi, bilingui EN/FR) — senza un filtro, ogni articolo veniva estratto due
volte con lo stesso numero (personal_protection: 55 numeri su 61 ripetuti nel
solo corpo).

Il meccanismo e' generico (``foreign_marker_ratio``, presa un pattern
esplicito): qui si carica il profilo EN direttamente con ``read_profile_file``,
come fa ``test_language_en.py``, cosi' il test non dipende dalla lingua attiva
nel processo che esegue pytest.
"""

from __future__ import annotations

import re

import pytest

from legisleaf.language import bundled_config_path, read_profile_file
from legisleaf.f2.build_tree import foreign_marker_ratio


@pytest.fixture(scope="module")
def en_marker_re() -> re.Pattern[str]:
    profile = read_profile_file(bundled_config_path("en"), origin="test")
    pattern = profile.convention("foreign_text_marker_pattern")
    assert pattern, "language_en.json deve dichiarare foreign_text_marker_pattern"
    return re.compile(pattern, re.IGNORECASE)


def test_paragrafo_francese_supera_la_soglia(en_marker_re):
    testo = (
        "Incompatibilité — lois. En cas d'incompatibilité entre les dispositions "
        "d'une loi codificatrice et celles de la loi d'origine, les dispositions "
        "de la loi d'origine avec ses modifications l'emportent dans la mesure "
        "de l'incompatibilité."
    )
    assert foreign_marker_ratio(testo, en_marker_re) >= 0.03


def test_paragrafo_inglese_resta_sotto_soglia(en_marker_re):
    testo = (
        "In the event of an inconsistency between a consolidation and the "
        "original statute or regulation as amended, the original statute or "
        "regulation as amended prevails to the extent of the inconsistency."
    )
    assert foreign_marker_ratio(testo, en_marker_re) < 0.03


def test_nomi_propri_con_diacritico_non_bastano(en_marker_re):
    """Caso reale trovato su data_protection: un paragrafo tutto in inglese
    che cita ripetutamente istituzioni irlandesi ('Dail Eireann', 'Seanad
    Eireann', nomi propri con diacritico) non deve superare la soglia — il
    diacritico su un nome proprio non rende straniera la frase che lo
    contiene."""
    testo = (
        "In this section, 'Committee' means a Committee appointed by either "
        "House of the Oireachtas or jointly by both Houses, including the "
        "Committee established jointly by Dáil Éireann and Seanad Éireann "
        "known as the Committee on Justice and Equality or any Committee "
        "established to replace that Committee."
    )
    assert foreign_marker_ratio(testo, en_marker_re) < 0.03


def test_nessun_pattern_non_scarta_mai():
    """Un profilo che non dichiara il marcatore (es. quello italiano, che non
    lo definisce in structure_conventions) lascia la funzione disattiva:
    ratio 0 qualunque sia il testo."""
    assert foreign_marker_ratio("Incompatibilité entre les dispositions", None) == 0.0
