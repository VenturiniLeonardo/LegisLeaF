"""Test della confidenza ricavata dai logprobs (f1b).

f1b chiedeva i logprobs al server e poi ritornava ``None`` incondizionatamente:
la soglia ``CONFIDENCE_ACCEPT_THRESHOLD`` non poteva mai essere superata e ogni
blocco passava dalla revisione LLM. Sui cinque documenti del corpus inglese
significava 93-99% dei blocchi rivisti dal modello grande.

I test coprono le forme in cui il backend puo' rispondere, incluse quelle in cui
la confidenza NON e' calcolabile: in quei casi il valore deve restare ``None``,
cioe' il comportamento prudente di prima, non un numero inventato.
"""

from __future__ import annotations

import math
import sys
from types import ModuleType, SimpleNamespace

import pytest

# ``f1b`` importa faiss, sentence-transformers e il client OpenAI, che vivono
# nell'env della pipeline e non servono a questa funzione: e' aritmetica sui
# logprobs, senza dipendenze. Si sostituiscono con segnaposto per rendere il test
# eseguibile anche dove l'env pesante non c'e', invece di saltarlo e non
# verificarlo mai.
for _nome in ("faiss", "sentence_transformers", "openai", "tqdm", "tqdm.auto"):
    if _nome not in sys.modules:
        _modulo = ModuleType(_nome)
        _modulo.__getattr__ = lambda _attr: object  # type: ignore[attr-defined]
        sys.modules[_nome] = _modulo

from legisleaf.f1.classify_blocks_slm import (  # noqa: E402
    label_confidence_from_logprobs,
)


def risposta(token_logprob: list[tuple[str, float]] | None) -> SimpleNamespace:
    """Risposta OpenAI-compatible con o senza blocco logprobs."""
    if token_logprob is None:
        return SimpleNamespace(choices=[SimpleNamespace(logprobs=None)])
    contenuto = [SimpleNamespace(token=t, logprob=lp) for t, lp in token_logprob]
    return SimpleNamespace(choices=[SimpleNamespace(logprobs=SimpleNamespace(content=contenuto))])


def test_etichetta_in_un_solo_token():
    r = risposta([('{"etichetta": "', -0.001), ("article", -0.02), ('"}', -0.001)])
    conf = label_confidence_from_logprobs(r, "article")
    assert conf == pytest.approx(math.exp(-0.02), rel=1e-9)


def test_etichetta_spezzata_in_piu_token():
    """"unresolved" arriva come 'un' + 'resolved': conta la media per token."""
    r = risposta([('{"etichetta": "', -0.001), ("un", -0.1), ("resolved", -0.2), ('"}', -0.001)])
    conf = label_confidence_from_logprobs(r, "unresolved")
    assert conf == pytest.approx(math.exp(-0.15), rel=1e-9)


def test_etichette_di_lunghezza_diversa_sono_confrontabili():
    """Il difetto che la media geometrica evita: penalizzare la tokenizzazione.

    Un token al 97% e due token al 97% descrivono la stessa certezza per token;
    col prodotto il secondo caso darebbe 0.94 e cadrebbe sotto soglia per la sola
    lunghezza dell'etichetta.
    """
    import math as _m

    lp = _m.log(0.97)
    uno = risposta([('{"etichetta": "', -0.0), ("content", lp), ('"}', -0.0)])
    due = risposta([('{"etichetta": "', -0.0), ("un", lp), ("resolved", lp), ('"}', -0.0)])
    assert label_confidence_from_logprobs(uno, "content") == pytest.approx(
        label_confidence_from_logprobs(due, "unresolved"), rel=1e-9
    )


def test_le_graffe_e_il_nome_del_campo_non_diluiscono():
    """Lo schema strict impone la cornice: includerla gonfierebbe o abbasserebbe a caso."""
    cornice_certa = risposta([('{"etichetta": "', -0.0), ("point", -0.5), ('"}', -0.0)])
    cornice_incerta = risposta([('{"etichetta": "', -3.0), ("point", -0.5), ('"}', -2.0)])
    assert label_confidence_from_logprobs(cornice_certa, "point") == pytest.approx(
        label_confidence_from_logprobs(cornice_incerta, "point"), rel=1e-9
    )


def test_confidenza_alta_per_decisione_netta():
    r = risposta([('{"etichetta": "', -0.0), ("annex", -0.0001), ('"}', -0.0)])
    assert label_confidence_from_logprobs(r, "annex") > 0.99


def test_confidenza_bassa_per_decisione_incerta():
    r = risposta([('{"etichetta": "', -0.0), ("content", -1.6), ('"}', -0.0)])
    conf = label_confidence_from_logprobs(r, "content")
    assert 0.15 < conf < 0.25


def test_il_riconoscimento_deterministico_supera_la_soglia():
    """Un match di regex non deve essere rimandato a un modello."""
    from legisleaf.common import load_config
    from legisleaf.f1.classify_blocks_slm import DETERMINISTIC_STRUCTURAL_CONFIDENCE

    assert DETERMINISTIC_STRUCTURAL_CONFIDENCE > load_config().confidence_accept_threshold


# --------------------------------------------------------------------------- #
# Casi in cui la confidenza non e' calcolabile: deve restare None              #
# --------------------------------------------------------------------------- #
def test_backend_senza_logprobs():
    assert label_confidence_from_logprobs(risposta(None), "article") is None


def test_blocco_logprobs_vuoto():
    assert label_confidence_from_logprobs(risposta([]), "article") is None


def test_etichetta_non_localizzabile_nei_token():
    """Tokenizzazione byte-level o testo non ricostruibile: nessuna invenzione."""
    r = risposta([("\\ufffd", -0.1), ("\\ufffd", -0.2)])
    assert label_confidence_from_logprobs(r, "article") is None


def test_logprob_assente_su_un_token_dell_etichetta():
    r = risposta([('{"etichetta": "', -0.001), ("art", -0.1), ("icle", None), ('"}', -0.001)])
    assert label_confidence_from_logprobs(r, "article") is None


def test_prende_l_ultima_occorrenza_dell_etichetta():
    """Se il nome dell'etichetta compare anche nel prompt riflesso, conta il valore."""
    r = risposta([("article", -2.0), (': {"etichetta": "', -0.001), ("article", -0.05), ('"}', -0.001)])
    assert label_confidence_from_logprobs(r, "article") == pytest.approx(math.exp(-0.05), rel=1e-9)


def test_soglia_di_accettazione_e_ora_raggiungibile():
    """Il difetto originale: nessun valore poteva superare CONFIDENCE_ACCEPT_THRESHOLD."""
    from legisleaf.common import load_config

    soglia = load_config().confidence_accept_threshold
    netta = risposta([('{"etichetta": "', -0.0), ("article", -0.01), ('"}', -0.0)])
    assert label_confidence_from_logprobs(netta, "article") > soglia
