"""Test della baseline inglese.

Fissano le due decisioni che rendono utilizzabile il profilo, e che i profili
generati dall'LLM sbagliavano:

* ``article`` e' l'unita' normativa foglia del common law (``Section 5``,
  ``§ 164.102``), non la parola "Article" — che in uno statuto britannico,
  irlandese o statunitense non compare mai;
* il ``(N)`` nudo NON e' un considerando: nel common law e' il marcatore
  standard di sottosezione, e riconoscerlo come recital faceva sparire 79 nodi
  su 133 da un atto irlandese e 32 su 34 da una public law.
"""

from __future__ import annotations

import pytest

from legisleaf.language import bundled_config_path, read_profile_file, validate_payload

STRUCTURAL_ORDER = ("annex", "article", "part", "title", "chapter", "section", "recital")


@pytest.fixture(scope="module")
def profile():
    return read_profile_file(bundled_config_path("en"), origin="test")


def riconosce(profile, testo: str) -> str | None:
    """Prima etichetta che fa presa, nell'ordine usato da infer_structural_label."""
    return next((label for label in STRUCTURAL_ORDER if profile.header_patterns[label].match(testo)), None)


def test_baseline_valida():
    import json

    validate_payload(json.loads(bundled_config_path("en").read_text(encoding="utf-8")), where="baseline en")


@pytest.mark.parametrize(
    "testo, atteso",
    [
        ("Section 5", "article"),
        ("Sec. 12", "article"),
        ("§ 164.102", "article"),
        ("Article 12", "article"),
        ("Section 12A", "article"),
        ("Section 164.312 Technical safeguards.", "article"),
        ("Part 1", "part"),
        ("PART I", "part"),
        ("Title 45", "title"),
        ("Chapter 2", "chapter"),
        ("Subpart A", "section"),
        ("Subpart E", "section"),
        ("Subchapter II", "section"),
        ("Schedule 1", "annex"),
        ("Annex II", "annex"),
        ("Appendix A", "annex"),
        ("Whereas:", "recital"),
    ],
)
def test_intestazioni_riconosciute(profile, testo, atteso):
    assert riconosce(profile, testo) == atteso


@pytest.mark.parametrize(
    "testo",
    [
        "(1) ensuring the highest level of cybersecurity at agencies",
        "(2) This Act and the Data Protection Acts 1988 and 2003 may be cited together",
        "(a) the provider of the service",
        "(i) any personal data breach",
        "This Act shall come into operation on such day as the Minister may appoint",
    ],
)
def test_le_sottosezioni_non_sono_struttura(profile, testo):
    """Il caso che rompeva il corpus: (N) e' un comma, non un considerando."""
    assert riconosce(profile, testo) is None


def test_article_e_section_non_si_contendono_le_stesse_righe(profile):
    """``Section 5`` deve essere article e SOLO article.

    Nei profili generati dall'LLM ``article`` e ``section`` matchavano entrambi
    "Section N": vinceva article per ordine di prova, e il livello contenitore
    spariva. Qui ``section`` e' il livello Subpart/Subchapter e la collisione e'
    impossibile per costruzione.
    """
    assert profile.header_patterns["article"].match("Section 5")
    assert not profile.header_patterns["section"].match("Section 5")
    assert profile.header_patterns["section"].match("Subpart A")
    assert not profile.header_patterns["article"].match("Subpart A")


def test_numero_dell_articolo_conserva_il_suffisso_letterale(profile):
    """``section 12`` e ``section 12A`` sono due articoli diversi."""
    match = profile.patterns["article_components"].match("Section 12A Overview")
    assert match and match.group("number") == "12A"
    assert profile.patterns["article_components"].match("Section 12 Overview").group("number") == "12"


def test_numero_puntato_del_cfr(profile):
    match = profile.patterns["article_components"].match("§ 164.312 Technical safeguards.")
    assert match and match.group("number") == "164.312"


@pytest.mark.parametrize(
    "incipit, subtype, country",
    [
        ("Public Law 116-207", "public-law", "us"),
        ("Executive Order 14028", "executive-order", "us"),
        ("45 CFR Part 164", "cfr", "us"),
        ("Number 7 of 2018", "act", "ie"),
        ("Data Protection Act 2018", "act", "uk"),
        ("Regulation (EU) 2016/679", "regulation", "eu"),
    ],
)
def test_tipo_di_atto_riconosciuto(profile, incipit, subtype, country):
    for pattern, rule in profile.document_type_rules():
        if pattern.search(incipit.lower()):
            assert rule["subtype"] == subtype
            assert rule["country"] == country
            return
    pytest.fail(f"nessuna regola ha riconosciuto {incipit!r}")


def test_marcatori_di_comma_e_di_punto(profile):
    assert profile.patterns["paragraph_marker"].match("(1) The controller shall")
    assert profile.patterns["paragraph_marker"].match("(3A) In this section")
    assert profile.patterns["point_marker"].match("(a) the provider")
    assert profile.patterns["point_marker"].match("(iv) any other person")
    assert not profile.patterns["paragraph_marker"].match("(a) the provider")
