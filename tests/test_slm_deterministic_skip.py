"""Test del salto deterministico e del parallelismo di f1b.

Prima f1b interrogava il modello per ogni blocco tranne quelli riconosciuti come
intestazione strutturale: il 9.2% del corpus. Comma numerato, punto di elenco e
numero di pagina nudo sono altrettanto deterministici e portano la quota al
57.8%, con un'etichetta che coincide con quella finale della pipeline nel 93.4%
dei casi.

I test fissano quali forme possono saltare il modello e — soprattutto — quali
NON devono: un blocco di testo discorsivo e un ``recovered_extra_candidate``
vanno sempre classificati dal modello.
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

# f1b importa faiss, sentence-transformers e il client OpenAI: qui non servono.
for _nome in ("faiss", "sentence_transformers", "openai", "tqdm", "tqdm.auto"):
    if _nome not in sys.modules:
        _modulo = ModuleType(_nome)
        _modulo.__getattr__ = lambda _attr: object  # type: ignore[attr-defined]
        sys.modules[_nome] = _modulo

from legisleaf.f1 import classify_blocks_slm as f1b  # noqa: E402


@pytest.mark.parametrize(
    "testo, atteso, motivo",
    [
        ("Art. 5 Definizioni", "article", "deterministic_structural_header"),
        ("Capo IV", "chapter", "deterministic_structural_header"),
        ("Allegato II", "annex", "deterministic_structural_header"),
        ("1. Il titolare adotta misure adeguate.", "paragraph", "deterministic_paragraph_marker"),
        ("a) i fornitori di servizi digitali;", "point", "deterministic_point_marker"),
        ("42", "noise", "deterministic_page_number"),
    ],
)
def test_forme_che_saltano_il_modello(testo, atteso, motivo):
    etichetta, ragione = f1b.deterministic_label_for({}, testo)
    assert etichetta == atteso
    assert ragione == motivo


@pytest.mark.parametrize(
    "testo",
    [
        "Il presente decreto disciplina il trattamento dei dati personali.",
        "Ai fini del presente atto si applicano le definizioni seguenti.",
        "IL PARLAMENTO E IL CONSIGLIO DELL'UNIONE EUROPEA",
    ],
)
def test_il_testo_discorsivo_va_sempre_al_modello(testo):
    assert f1b.deterministic_label_for({}, testo) == (None, None)


def test_i_blocchi_recuperati_vanno_sempre_al_modello():
    """``recovered_extra_candidate`` viene da un recupero incerto dell'estrazione."""
    block = {"recovered_extra_candidate": True}
    assert f1b.deterministic_label_for(block, "Art. 5 Definizioni") == (None, None)
    assert f1b.deterministic_label_for(block, "1. Il titolare adotta misure.") == (None, None)


def test_lo_switch_riporta_al_solo_salto_strutturale(monkeypatch):
    """Con DETERMINISTIC_CONTENT_MARKERS=0 resta il comportamento precedente."""
    monkeypatch.setattr(f1b, "DETERMINISTIC_CONTENT_MARKERS", False)
    assert f1b.deterministic_label_for({}, "Art. 5 Definizioni")[0] == "article"
    assert f1b.deterministic_label_for({}, "1. Il titolare adotta misure.") == (None, None)
    assert f1b.deterministic_label_for({}, "a) i fornitori;") == (None, None)


def test_la_confidenza_deterministica_evita_la_revisione_llm():
    """Un match di regex non deve essere rimandato a un modello per conferma."""
    from legisleaf.common import load_config

    assert f1b.DETERMINISTIC_STRUCTURAL_CONFIDENCE > load_config().confidence_accept_threshold


def test_il_parallelismo_e_configurabile_e_disattivabile(monkeypatch):
    """SLM_MAX_WORKERS=1 deve restare una via d'uscita esplicita."""
    import importlib

    monkeypatch.setenv("SLM_MAX_WORKERS", "1")
    ricaricato = importlib.reload(f1b)
    assert ricaricato.SLM_MAX_WORKERS == 1
    monkeypatch.setenv("SLM_MAX_WORKERS", "8")
    ricaricato = importlib.reload(f1b)
    assert ricaricato.SLM_MAX_WORKERS == 8
    monkeypatch.delenv("SLM_MAX_WORKERS")
    importlib.reload(f1b)
