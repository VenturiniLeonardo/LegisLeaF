"""Test del profilo linguistico della pipeline.

Coprono le tre cose che, se si rompono, si rompono in silenzio: l'espansione dei
segnaposto nei pattern, il rifiuto di un profilo non valido prima che venga
scritto su disco, e la sopravvivenza di una lingua senza suffissi ordinali.
"""

from __future__ import annotations

import json
import re

import pytest

from legisleaf.language import (
    ALLOWED_LABELS,
    SCHEMA_VERSION,
    LanguageProfileError,
    build_profile,
    bundled_config_path,
    load_language_profile,
    read_profile_file,
    validate_payload,
)


@pytest.fixture()
def italian_payload() -> dict:
    return json.loads(bundled_config_path("it").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Baseline italiana                                                            #
# --------------------------------------------------------------------------- #
def test_baseline_italiana_valida(italian_payload):
    validate_payload(italian_payload, where="baseline it")


def test_baseline_italiana_riconosce_le_intestazioni():
    profile = read_profile_file(bundled_config_path("it"), origin="test")
    casi = {
        "Art. 5-ter. Definizioni": "article",
        "Articolo 1": "article",
        "Capo IV": "chapter",
        "TITOLO I": "title",
        "Sezione 3": "section",
        "Allegato II-bis": "annex",
        "Parte PRIMA": "part",
    }
    for testo, etichetta in casi.items():
        assert profile.header_patterns[etichetta].match(testo), f"{testo!r} non riconosciuto come {etichetta}"


def test_regole_tipo_atto_ripiegano_sulla_baseline(italian_payload):
    """Un profilo scritto da f0 prima di ``akn_document_types`` resta valido.

    Le parole chiave del tipo di atto sono proprieta' della lingua, non del
    documento: senza il ripiego, ogni run gia' fatta perderebbe il
    riconoscimento finche' non si rigenera il profilo con l'LLM.
    """
    senza_sezione = {k: v for k, v in italian_payload.items() if k != "akn_document_types"}
    validate_payload(senza_sezione, where="profilo senza akn_document_types")

    profilo = build_profile(senza_sezione, origin="f0")
    assert profilo.akn_document_types == []
    assert [rule.get("subtype") for _, rule in profilo.document_type_rules()] == [
        rule.get("subtype") for rule in italian_payload["akn_document_types"]
    ]


@pytest.mark.parametrize(
    "consegnato",
    [
        [{"from": "Bis", "to": "bis"}],
        {"Bis": "bis"},
        [["Bis", "bis"]],
        ["Bis->bis"],
        ["Bis = bis"],
        ["Bis: bis"],
    ],
)
def test_suffix_aliases_tollera_le_forme_del_modello(consegnato):
    """Il server non onora sempre lo schema strict: la forma sbagliata non deve uccidere il batch."""
    from legisleaf.f0.detect_language import coerce_suffix_aliases

    assert coerce_suffix_aliases(consegnato) == {"bis": "bis"}


@pytest.mark.parametrize("consegnato", [None, [], ["senza-separatore"], [{"from": "bis"}], [42]])
def test_suffix_aliases_scarta_le_voci_inutilizzabili(consegnato):
    from legisleaf.f0.detect_language import coerce_suffix_aliases

    assert coerce_suffix_aliases(consegnato) == {}


def test_suffissi_ordinali_dal_piu_lungo_al_piu_corto():
    """Nelle alternazioni "ter" non deve mai vincere su "terdecies"."""
    profile = read_profile_file(bundled_config_path("it"), origin="test")
    lunghezze = [len(s) for s in profile.ordinal_suffixes]
    assert lunghezze == sorted(lunghezze, reverse=True)
    match = profile.patterns["article_components"].match("Art. 67-terdecies")
    assert match and match.group("suffix") == "terdecies"


# --------------------------------------------------------------------------- #
# Espansione dei segnaposto                                                    #
# --------------------------------------------------------------------------- #
def test_quantificatori_regex_non_vanno_raddoppiati(italian_payload):
    """``{1,4}`` resta letterale, ``{rubric_max}`` viene sostituito."""
    profile = build_profile(italian_payload, origin="test")
    assert profile.patterns["page_number_noise"].pattern == r"\d{1,4}|Pag\.?\s*\d{1,4}"
    assert "{rubric_max}" not in profile.header_patterns["article"].pattern
    assert "{1,160}" in profile.header_patterns["article"].pattern


def test_segnaposto_sconosciuto_resta_letterale(italian_payload):
    """Un segnaposto non previsto non deve far esplodere il caricamento.

    Deve arrivare invariato a ``re.compile``, che lo rifiutera' con un errore
    esplicito se non e' una regex valida, invece di sparire silenziosamente.
    """
    payload = json.loads(json.dumps(italian_payload))
    payload["patterns"]["annex_local_part"] = r"^Parte\s+\d+\s*{sconosciuto}?$"
    profile = build_profile(payload, origin="test")
    assert "{sconosciuto}" in profile.patterns["annex_local_part"].pattern


# --------------------------------------------------------------------------- #
# Validazione                                                                  #
# --------------------------------------------------------------------------- #
def test_rifiuta_pattern_non_compilabile(italian_payload):
    payload = json.loads(json.dumps(italian_payload))
    payload["header_patterns"]["chapter"] = r"^Capo\s+([IVXLCDM]+"
    with pytest.raises(LanguageProfileError, match="non compilabile"):
        validate_payload(payload)


def test_rifiuta_header_pattern_mancante(italian_payload):
    payload = json.loads(json.dumps(italian_payload))
    payload["header_patterns"].pop("article")
    with pytest.raises(LanguageProfileError, match="header_patterns mancanti"):
        validate_payload(payload)


def test_rifiuta_pattern_di_servizio_mancante(italian_payload):
    payload = json.loads(json.dumps(italian_payload))
    payload["patterns"].pop("device_start")
    with pytest.raises(LanguageProfileError, match="patterns mancanti"):
        validate_payload(payload)


def test_rifiuta_etichette_inventate(italian_payload):
    """Le etichette sono chiavi di programma: tradurle rompe tutta la pipeline."""
    payload = json.loads(json.dumps(italian_payload))
    payload["label_descriptions"]["articolo"] = payload["label_descriptions"].pop("article")
    with pytest.raises(LanguageProfileError):
        validate_payload(payload)


def test_rifiuta_schema_version_diversa(italian_payload):
    payload = json.loads(json.dumps(italian_payload))
    payload["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(LanguageProfileError, match="schema_version"):
        validate_payload(payload)


def test_label_descriptions_copre_tutte_le_etichette(italian_payload):
    assert set(italian_payload["label_descriptions"]) == ALLOWED_LABELS


# --------------------------------------------------------------------------- #
# Lingue senza suffissi ordinali                                               #
# --------------------------------------------------------------------------- #
def test_lingua_senza_suffissi_ordinali(italian_payload):
    """Con ``ordinal_suffixes`` vuoto il gruppo del suffisso deve esistere e non matchare.

    ``extract_structural_number`` legge sempre il gruppo 2 degli header pattern:
    se sparisse, la lettura del numero andrebbe in IndexError; se matchasse a
    vuoto, divorerebbe il testo che segue il numero.
    """
    payload = json.loads(json.dumps(italian_payload))
    payload["ordinal_suffixes"] = []
    payload["suffix_aliases"] = {}
    validate_payload(payload)

    profile = build_profile(payload, origin="test")
    match = profile.header_patterns["chapter"].match("Capo IV")
    assert match is not None
    assert match.group(1) == "IV"
    assert match.group(2) is None


def test_lingua_senza_numerali_a_parola(italian_payload):
    payload = json.loads(json.dumps(italian_payload))
    payload["numbering"]["word_numerals"] = []
    profile = build_profile(payload, origin="test")
    # Senza numerali a parola il gruppo non deve diventare "(?:|{num})", che
    # matcherebbe la stringa vuota e riconoscerebbe "Parte" come partizione.
    assert profile.part_number_pattern == profile.roman_or_arabic_pattern
    assert profile.header_patterns["part"].match("Parte II")
    assert not profile.header_patterns["part"].match("Parte del contratto")


# --------------------------------------------------------------------------- #
# Risoluzione del profilo attivo                                               #
# --------------------------------------------------------------------------- #
def test_profilo_del_documento_vince_sulla_baseline(tmp_path, italian_payload, monkeypatch):
    payload = json.loads(json.dumps(italian_payload))
    payload["code"] = "xx"
    payload["name"] = "finta"
    (tmp_path / "language_config.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.delenv("LANGUAGE_CONFIG_PATH", raising=False)
    profile = load_language_profile(tmp_path)
    assert profile.code == "xx"
    assert profile.origin == "f0"


def test_senza_profilo_del_documento_si_usa_la_baseline(tmp_path, monkeypatch):
    monkeypatch.delenv("LANGUAGE_CONFIG_PATH", raising=False)
    monkeypatch.delenv("LANGUAGE_CODE", raising=False)
    profile = load_language_profile(tmp_path)
    assert profile.code == "it"
    assert profile.origin == "baseline"


def test_lingua_senza_baseline_e_senza_f0_fallisce(tmp_path, monkeypatch):
    """Meglio un errore esplicito che processare un documento francese con regex italiane."""
    monkeypatch.delenv("LANGUAGE_CONFIG_PATH", raising=False)
    monkeypatch.setenv("LANGUAGE_CODE", "zz")
    with pytest.raises(LanguageProfileError, match="f0"):
        load_language_profile(tmp_path)


def test_impronta_ignora_i_metadati_di_riconoscimento(italian_payload):
    """Rieseguire f0 non deve invalidare la cache di f1a se il profilo non cambia."""
    a = build_profile(json.loads(json.dumps(italian_payload)), origin="test")
    con_detection = json.loads(json.dumps(italian_payload))
    con_detection["detection"] = {"code": "it", "confidence": 0.99}
    con_detection["source"] = "generato-da-f0"
    b = build_profile(con_detection, origin="test")
    assert a.fingerprint()["sha256"] == b.fingerprint()["sha256"]


# --------------------------------------------------------------------------- #
# Parita' con le regex storiche della pipeline                                 #
# --------------------------------------------------------------------------- #
def test_f1_common_usa_il_profilo_attivo():
    from legisleaf import common as f1_common

    assert f1_common.LANGUAGE.code == "it"
    assert f1_common.HEADER_PATTERNS is f1_common.LANGUAGE.header_patterns
    assert f1_common.is_article_header("Art. 5-ter. Definizioni")
    assert f1_common.extract_structural_number("Capo IV", "chapter") == "IV"
    assert f1_common.extract_structural_number("Art. 5 bis", "article") == "5-bis"
    assert f1_common.is_modificative_instruction("Dopo l'articolo 3 e' inserito il capo seguente")
    assert f1_common.infer_structural_label("Allegato II-bis") == "annex"


def test_dash_pattern_copre_i_trattini_tipografici():
    from legisleaf.language import DASH_PATTERN

    for dash in "-‐‑‒–—―":
        assert re.fullmatch(DASH_PATTERN, dash), f"trattino non coperto: {dash!r}"
