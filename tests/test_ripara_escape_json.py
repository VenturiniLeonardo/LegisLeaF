"""Riparazione degli escape JSON nelle risposte di f1a.

Il caso che ha motivato questi test e' reale: il 9/9/2026 la run su HRDoc si e'
fermata su ``arxiv_1401_8087`` perche' il modello, ricopiando LaTeX dentro una
stringa JSON, aveva prodotto ``$\\begin{array}...\\\\ \\end{array}$``. La
riparazione di allora scandiva un backslash alla volta e su ``\\\\ `` (due
backslash veri seguiti da spazio) ne produceva TRE, cioe' una coppia valida piu'
un ``\\ `` di nuovo illegale: rompeva invece di riparare, e con --phase-first
bastava quel documento per abortire l'intero corpus.

Qui si fissa il contratto: dopo la riparazione il JSON deve essere sempre
parsabile, e cio' che era gia' valido non deve essere toccato.
"""

import json

from legisleaf.f1.prepare_analysis import ripara_escape_json


def _parsa(testo: str) -> dict:
    return json.loads(ripara_escape_json(testo))


def test_a_capo_latex_seguito_da_spazio():
    """Il caso di arxiv_1401_8087: due backslash veri e poi uno spazio."""
    grezzo = r'{"t": "$\begin{array}{l}{H(x,y)}\\ \end{array}$"}'
    assert _parsa(grezzo)["t"].endswith("$")


def test_escape_non_valido_singolo():
    assert _parsa(r'{"t": "\alpha e \gamma"}')["t"]


def test_sequenze_dispari_lunghe():
    """Tre backslash prima di una lettera non valida restano pari dopo."""
    assert _parsa(r'{"t": "\\\gamma"}')["t"]


def test_u_senza_quattro_esadecimali():
    """``\\usepackage`` non e' un escape unicode e va riparato."""
    assert "usepackage" in _parsa(r'{"t": "\usepackage{amsmath}"}')["t"]


def test_unicode_valido_non_toccato():
    assert _parsa(r'{"t": "à"}')["t"] == "à"


def test_escape_validi_non_toccati():
    """Cio' che era gia' corretto deve restare identico: se la riparazione
    aggiungesse un backslash qui, cambierebbe il testo degli esempi."""
    for grezzo in (r'{"t": "riga\nriga"}', r'{"t": "coppia \\ sola"}',
                   r'{"t": "con \"virgolette\" dentro"}'):
        assert ripara_escape_json(grezzo) == grezzo
        json.loads(grezzo)


def test_json_gia_valido_invariato():
    grezzo = '{"golden_set": [], "label_rilevanti": ["article"]}'
    assert ripara_escape_json(grezzo) == grezzo
