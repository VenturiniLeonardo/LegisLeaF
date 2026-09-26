"""F0 - Riconoscimento della lingua del documento e scrittura del profilo linguistico.

La pipeline nasce su testi normativi italiani: parole chiave ("CAPO", "Art.",
"considerando quanto segue"), suffissi ordinali latini ed esempi dei prompt erano
scritti dentro il codice degli step. Su un documento inglese o francese ogni
regex strutturale falliva in silenzio: nessun nodo, albero piatto, e nessun
errore a segnalarlo.

Questo step gira PRIMA di f1a e fa due cose:

1. riconosce la lingua del documento (il PDF da cui proviene l'OCR in ingresso),
   con un LLM e una controprova deterministica su stopword;
2. scrive ``<OUTPUT_DIR>/language_config.json``, il profilo linguistico che tutti
   gli step successivi leggono al posto dei letterali italiani.

Se per la lingua riconosciuta esiste una baseline versionata in
``legisleaf/config/language_<codice>.json`` viene usata quella: e' il
profilo tarato e testato, e rigenerarlo con un LLM significherebbe sostituire
regex verificate con regex plausibili. Solo per le lingue senza baseline il
profilo viene generato dal modello grande (nemotron-3-super) e validato prima di
essere scritto: un profilo non compilabile viene rifiutato qui, non tre step piu'
avanti.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from openai import OpenAI

from legisleaf.language import (
    ALLOWED_LABELS,
    SCHEMA_VERSION,
    LanguageProfileError,
    active_config_path,
    build_profile,
    bundled_config_path,
    bundled_language_codes,
    validate_payload,
)
from legisleaf.language.profile import (
    REQUIRED_DISPLAY_KEYS,
    REQUIRED_HEADER_LABELS,
    REQUIRED_PATTERNS,
)
from legisleaf.common import F1Config, load_blocks, load_config, write_json

MAX_SAMPLE_CHARS = int(os.getenv("F0_MAX_SAMPLE_CHARS", "6000"))
HEAD_BLOCKS = int(os.getenv("F0_HEAD_BLOCKS", "40"))
SPREAD_BLOCKS = int(os.getenv("F0_SPREAD_BLOCKS", "60"))
SHORT_LINE_BLOCKS = int(os.getenv("F0_SHORT_LINE_BLOCKS", "40"))
SHORT_LINE_MAX_CHARS = int(os.getenv("F0_SHORT_LINE_MAX_CHARS", "80"))
USE_CACHED_PROFILE = os.getenv("F0_USE_CACHED_PROFILE", "1") == "1"
ALWAYS_GENERATE = os.getenv("F0_ALWAYS_GENERATE", "0") == "1"
FORCED_LANGUAGE = (os.getenv("F0_FORCE_LANGUAGE") or "").strip().lower()
MIN_STRUCTURAL_HITS = int(os.getenv("F0_MIN_STRUCTURAL_HITS", "1"))

DETECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string"},
        "name": {"type": "string"},
        "english_name": {"type": "string"},
        "confidence": {"type": "number"},
        "evidence": {"type": "string"},
    },
    "required": ["code", "name", "english_name", "confidence", "evidence"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------- #
# Campione di testo                                                            #
# --------------------------------------------------------------------------- #
def build_language_sample(blocks: list[dict]) -> str:
    """Campione per il riconoscimento: testa, prelievo uniforme e righe brevi.

    Le tre fonti servono a cose diverse. La testa (frontespizio, preambolo, prime
    intestazioni) e' dove la lingua si vede meglio. Il prelievo uniforme evita che
    un documento con copertina bilingue o allegati in un'altra lingua venga
    classificato sulle sole prime pagine. Le righe brevi sono quasi sempre
    intestazioni: senza di loro un atto UE, che apre con decine di considerando,
    darebbe un campione in cui la parola "Articolo" non compare mai — e i pattern
    strutturali del profilo non sarebbero verificabili.
    """
    texts = [block["testo"] for block in blocks if (block.get("testo") or "").strip()]
    if not texts:
        raise ValueError("Nessun blocco testuale nel documento: impossibile riconoscere la lingua.")

    head = texts[:HEAD_BLOCKS]
    rest = texts[HEAD_BLOCKS:]

    spread: list[str] = []
    if rest and SPREAD_BLOCKS > 0:
        step = max(1, len(rest) // SPREAD_BLOCKS)
        spread = rest[::step][:SPREAD_BLOCKS]

    # Criterio volutamente indipendente dalla lingua: una riga corta con almeno
    # una lettera e almeno una cifra e' un candidato intestazione in qualunque
    # nomenclatura ("CAPO IV", "Article 5", "Anhang II").
    short_candidates = [
        text
        for text in rest
        if len(text) <= SHORT_LINE_MAX_CHARS
        and re.search(r"\d", text)
        and re.search(r"[^\W\d_]", text, flags=re.UNICODE)
    ]
    if short_candidates and SHORT_LINE_BLOCKS > 0:
        step = max(1, len(short_candidates) // SHORT_LINE_BLOCKS)
        short_candidates = short_candidates[::step][:SHORT_LINE_BLOCKS]

    kept: list[str] = []
    seen: set[str] = set()
    used = 0
    for text in head + short_candidates + spread:
        if text in seen:
            continue
        if kept and used + len(text) + 1 > MAX_SAMPLE_CHARS:
            continue
        seen.add(text)
        kept.append(text)
        used += len(text) + 1
    return "\n".join(kept)


def _words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return re.findall(r"[^\W\d_]+", normalized, flags=re.UNICODE)


def deterministic_language_hint(sample: str) -> dict[str, Any]:
    """Controprova su stopword delle baseline disponibili.

    Non sostituisce l'LLM (con una sola baseline non saprebbe distinguere fra
    inglese e francese), ma smaschera il caso pericoloso: LLM che dichiara una
    lingua mentre il testo e' chiaramente un'altra fra quelle che conosciamo.
    """
    tokens = Counter(_words(sample))
    total = max(1, sum(tokens.values()))

    scores: dict[str, float] = {}
    for code in bundled_language_codes():
        payload = json.loads(bundled_config_path(code).read_text(encoding="utf-8"))
        hints = payload.get("detection_hints", {})

        stopwords = []
        for parola in hints.get("stopwords", []):
            stopwords.append(parola.lower())
        if not stopwords:
            continue

        # Quante occorrenze delle stopword di questa lingua compaiono nel
        # campione. Le voci con uno spazio ("ai sensi") non sono singoli token
        # e non si possono contare qui.
        hits = 0
        for parola in stopwords:
            if " " in parola:
                continue
            hits += tokens[parola]
        scores[code] = round(hits / total, 4)

    if not scores:
        return {"scores": scores, "best": None, "best_score": None}

    # Lingua con la quota di stopword piu' alta.
    best = None
    best_score = None
    for code, punteggio in scores.items():
        if best_score is None or punteggio > best_score:
            best = code
            best_score = punteggio
    return {"scores": scores, "best": best, "best_score": best_score}


# --------------------------------------------------------------------------- #
# Riconoscimento della lingua                                                  #
# --------------------------------------------------------------------------- #
def detect_language_with_llm(sample: str, config: F1Config) -> dict[str, Any]:
    client = OpenAI(base_url=config.openai_base_url, api_key=config.openai_api_key)
    response = client.chat.completions.create(
        model=config.model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "Riconosci la lingua principale di un testo normativo estratto via OCR. "
                    "Rispondi solo con JSON valido."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Indica la lingua PRINCIPALE del testo seguente.\n"
                    "- 'code': codice ISO 639-1 minuscolo (it, en, fr, de, es, ...).\n"
                    "- 'name': nome della lingua in italiano (italiano, inglese, francese, ...).\n"
                    "- 'english_name': nome della lingua in inglese.\n"
                    "- 'confidence': 0.0-1.0.\n"
                    "- 'evidence': fino a 15 parole del testo che giustificano la scelta.\n"
                    "Ignora citazioni isolate in altre lingue: conta la lingua del corpo normativo.\n\n"
                    f"--- INIZIO TESTO ---\n{sample}\n--- FINE TESTO ---"
                ),
            },
        ],
        temperature=0.0,
        top_p=1.0,
        seed=42,
        max_tokens=256,
        timeout=config.llm_timeout_seconds,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}, "top_k": 1},
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "language_detection", "strict": True, "schema": DETECTION_SCHEMA},
        },
    )
    detection = json.loads(response.choices[0].message.content)
    detection["code"] = detection["code"].strip().lower()[:5]
    return detection


# --------------------------------------------------------------------------- #
# Generazione del profilo per lingue senza baseline                            #
# --------------------------------------------------------------------------- #
def profile_generation_schema() -> dict[str, Any]:
    string = {"type": "string"}
    string_list = {"type": "array", "items": {"type": "string"}}

    def fixed_object(keys) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {key: string for key in keys},
            "required": list(keys),
            "additionalProperties": False,
        }

    return {
        "type": "object",
        "properties": {
            "code": string,
            "name": string,
            "english_name": string,
            "quotes": string,
            "numbering": {
                "type": "object",
                "properties": {
                    "roman_or_arabic": string,
                    "article_number": string,
                    "word_numerals": string_list,
                },
                "required": ["roman_or_arabic", "article_number", "word_numerals"],
                "additionalProperties": False,
            },
            "ordinal_suffixes": string_list,
            "suffix_aliases": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"from": string, "to": string},
                    "required": ["from", "to"],
                    "additionalProperties": False,
                },
            },
            "akn_patterns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "etichetta": {"type": "string", "enum": sorted(ALLOWED_LABELS)},
                        "pattern": string,
                        "priorita": {"type": "integer"},
                    },
                    "required": ["etichetta", "pattern", "priorita"],
                    "additionalProperties": False,
                },
            },
            "header_patterns": fixed_object(REQUIRED_HEADER_LABELS),
            "patterns": fixed_object(REQUIRED_PATTERNS),
            "label_descriptions": fixed_object(sorted(ALLOWED_LABELS)),
            "label_examples": {
                "type": "object",
                "properties": {label: string_list for label in sorted(ALLOWED_LABELS)},
                "required": sorted(ALLOWED_LABELS),
                "additionalProperties": False,
            },
            "prompt_terms": {
                "type": "object",
                "properties": {
                    "hierarchy_glosses": fixed_object(
                        ("part", "annex", "title", "chapter", "section", "article")
                    ),
                    "modificative_examples": string_list,
                    "article_citation_examples": string_list,
                    "forbidden_label_synonyms": string_list,
                },
                "required": [
                    "hierarchy_glosses",
                    "modificative_examples",
                    "article_citation_examples",
                    "forbidden_label_synonyms",
                ],
                "additionalProperties": False,
            },
            "display": fixed_object(REQUIRED_DISPLAY_KEYS),
            "smoke_queries": fixed_object(("graph_search", "vector_search")),
            "detection_hints": {
                "type": "object",
                "properties": {"stopwords": string_list},
                "required": ["stopwords"],
                "additionalProperties": False,
            },
        },
        "required": [
            "code",
            "name",
            "english_name",
            "quotes",
            "numbering",
            "ordinal_suffixes",
            "suffix_aliases",
            "akn_patterns",
            "header_patterns",
            "patterns",
            "label_descriptions",
            "label_examples",
            "prompt_terms",
            "display",
            "smoke_queries",
            "detection_hints",
        ],
        "additionalProperties": False,
    }


def generation_prompt(sample: str, detection: dict, reference: dict, feedback: str | None) -> str:
    reference_text = json.dumps(
        {k: v for k, v in reference.items() if not k.startswith("_") and k not in {"source", "schema_version"}},
        ensure_ascii=False,
        indent=2,
    )
    retry_note = (
        f"\nIL TENTATIVO PRECEDENTE E' STATO RIFIUTATO. Correggi esattamente questo problema:\n{feedback}\n"
        if feedback
        else ""
    )
    return f"""Devi adattare il profilo linguistico di una pipeline di estrazione strutturale
di testi normativi dalla lingua italiana alla lingua {detection['name']} (codice {detection['code']}).

Il profilo di riferimento in italiano, gia' collaudato, e' questo:

{reference_text}

REGOLE DI ADATTAMENTO
1. Mantieni ESATTAMENTE la stessa struttura di campi e le stesse chiavi.
2. Traduci solo cio' che dipende dalla lingua: parole chiave strutturali
   (PARTE/TITOLO/CAPO/SEZIONE/ALLEGATO/Articolo), verbi delle istruzioni
   modificative, marcatori di preambolo e di inizio articolato, prefissi di
   visualizzazione, descrizioni ed esempi delle etichette.
3. NON tradurre le etichette: restano in inglese Akoma Ntoso
   (part, annex, title, chapter, section, article, recital, paragraph, point,
   content, unresolved, noise). Sono chiavi di programma, non testo.
4. Le descrizioni in "label_descriptions" e le glosse in "hierarchy_glosses"
   restano scritte in ITALIANO (e' la lingua dei prompt della pipeline), ma gli
   ESEMPI dentro di esse devono essere nella lingua {detection['name']}.
5. "label_examples" contiene esempi copiati dalla forma reale del documento
   campione qui sotto, non traduzioni letterali dall'italiano.
6. Nei pattern usa i segnaposto {{num}}, {{article_num}}, {{part_number}},
   {{suffix}}, {{opt_suffix}}, {{dash}}, {{quotes}}, {{rubric_max}} esattamente
   come nel riferimento. Le graffe dei quantificatori regex si scrivono normali:
   {{1,4}} resta {{1,4}}, non va raddoppiata.
7. I gruppi di cattura devono restare nella stessa posizione del riferimento:
   header_patterns cattura il numero nel gruppo 1 e il suffisso nel gruppo 2
   (fornito da {{opt_suffix}}); "article_components" deve conservare i gruppi
   nominati prefix, number, suffix, rubrica; "recital_number" deve avere due
   gruppi alternativi come nel riferimento.
8. "ordinal_suffixes" elenca i suffissi che nella lingua {detection['name']}
   distinguono un articolo aggiunto (in italiano bis, ter, quater...). Se la
   lingua non ne usa, restituisci una lista vuota.
9. "suffix_aliases" e' una lista di coppie {{"from": ..., "to": ...}}; usa una
   lista vuota se la lingua non ha varianti da unificare.
10. "detection_hints.stopwords" elenca 10-20 parole molto frequenti nella lingua
    {detection['name']}, utili a riconoscerla.
11. "smoke_queries" contiene due query di prova nella lingua {detection['name']}:
    "graph_search" cerca un'intestazione di articolo nel grafo, "vector_search" e'
    un termine di dominio che nel documento campione compare di sicuro.
{retry_note}
Osserva la forma reale del documento prima di scrivere i pattern.

--- INIZIO CAMPIONE DEL DOCUMENTO ---
{sample}
--- FINE CAMPIONE ---

Restituisci SOLO il JSON del profilo adattato."""


ALIAS_SEPARATORS = ("->", "=>", "=", ":", "|")


def coerce_suffix_aliases(value: Any) -> dict[str, str]:
    """Riduce a dizionario le forme in cui il modello consegna ``suffix_aliases``.

    Lo schema chiede una lista di coppie ``{"from": ..., "to": ...}`` perche' un
    JSON schema strict non puo' descrivere un oggetto a chiavi libere. Il server
    pero' non sempre onora lo schema annidato: capita che consegni gia' un
    dizionario, oppure una lista di stringhe ("bis->bis", "quater: quater").
    Assumere la sola forma dichiarata faceva morire l'intero batch su un
    ``AttributeError``, invece di scartare la voce e ritentare.
    """
    if isinstance(value, dict):
        return {str(k).lower(): str(v).lower() for k, v in value.items() if k and v}

    aliases: dict[str, str] = {}
    for item in value or []:
        if isinstance(item, dict):
            source, target = item.get("from"), item.get("to")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            source, target = item
        elif isinstance(item, str):
            source = target = None
            for separator in ALIAS_SEPARATORS:
                if separator in item:
                    source, _, target = item.partition(separator)
                    break
        else:
            continue
        if source and target:
            aliases[str(source).strip().lower()] = str(target).strip().lower()
    return aliases


def normalize_generated_payload(raw: dict, detection: dict) -> dict:
    """Porta l'output del modello nella forma attesa dal loader."""
    payload = dict(raw)
    payload["schema_version"] = SCHEMA_VERSION
    payload["source"] = "generato-da-f0"
    payload["code"] = (payload.get("code") or detection["code"]).strip().lower()
    payload["article_rubric_max_chars"] = 160
    payload["suffix_aliases"] = coerce_suffix_aliases(raw.get("suffix_aliases"))
    return payload


def structural_coverage(payload: dict, sample: str) -> dict[str, int]:
    """Quante righe del campione ogni pattern di intestazione riconosce.

    Un profilo sintatticamente valido puo' essere comunque inutile: regex che non
    matchano nulla producono un albero piatto senza generare errori. La copertura
    e' l'unico controllo che distingue i due casi.
    """
    profile = build_profile(payload, origin="validation")
    lines = [line.strip() for line in sample.splitlines() if line.strip()]
    coverage = {}
    for label, pattern in profile.header_patterns.items():
        coverage[label] = sum(1 for line in lines if pattern.match(line))
    return coverage


def generate_profile_with_llm(sample: str, detection: dict, config: F1Config) -> tuple[dict, dict]:
    reference = json.loads(bundled_config_path("it").read_text(encoding="utf-8"))
    client = OpenAI(base_url=config.openai_base_url, api_key=config.openai_api_key)
    schema = profile_generation_schema()
    feedback: str | None = None

    for attempt in range(1, config.max_retries + 1):
        print(f"[f0] Generazione profilo '{detection['code']}' con {config.model_name} (tentativo {attempt}).")
        response = client.chat.completions.create(
            model=config.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Sei un ingegnere linguistico esperto di documenti normativi e di espressioni "
                        "regolari Python. Rispondi solo con JSON valido conforme allo schema."
                    ),
                },
                {"role": "user", "content": generation_prompt(sample, detection, reference, feedback)},
            ],
            temperature=0.0,
            top_p=1.0,
            seed=42,
            timeout=config.llm_timeout_seconds,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}, "top_k": 1},
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "language_profile", "strict": True, "schema": schema},
            },
        )
        # La normalizzazione sta DENTRO il try: il server non sempre onora lo
        # schema strict, e una chiave con il tipo sbagliato deve diventare
        # feedback per il tentativo successivo, non uccidere il batch.
        try:
            raw = json.loads(response.choices[0].message.content)
            payload = normalize_generated_payload(raw, detection)
            validate_payload(payload, where=f"profilo generato per '{detection['code']}'")
            coverage = structural_coverage(payload, sample)
        except (LanguageProfileError, re.error, json.JSONDecodeError, TypeError, AttributeError, ValueError) as exc:
            feedback = f"{exc.__class__.__name__}: {exc}"
            print(f"[f0] Profilo rifiutato: {feedback}")
            continue

        if sum(coverage.values()) < MIN_STRUCTURAL_HITS:
            feedback = (
                "I pattern di header_patterns non riconoscono NESSUNA riga del campione "
                f"(copertura {coverage}). Riscrivili osservando la forma reale delle intestazioni nel campione."
            )
            print(f"[f0] Profilo rifiutato: {feedback}")
            continue

        print(f"[f0] Profilo accettato. Copertura intestazioni sul campione: {coverage}")
        return payload, coverage

    raise RuntimeError(
        f"Impossibile generare un profilo linguistico valido per '{detection['code']}' "
        f"dopo {config.max_retries} tentativi. Ultimo errore: {feedback}"
    )


# --------------------------------------------------------------------------- #
# Cache                                                                        #
# --------------------------------------------------------------------------- #
def sample_signature(config: F1Config, sample: str) -> dict[str, Any]:
    return {
        "ocr_sha256": hashlib.sha256(config.blocks_json_path.read_bytes()).hexdigest()
        if config.blocks_json_path.exists()
        else None,
        "sample_sha256": hashlib.sha256(sample.encode("utf-8")).hexdigest(),
        "model_name": config.model_name,
        "forced_language": FORCED_LANGUAGE or None,
        "always_generate": ALWAYS_GENERATE,
        "schema_version": SCHEMA_VERSION,
    }


def load_cached_profile(path: Path, signature: dict) -> dict | None:
    if not USE_CACHED_PROFILE or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        print(f"[f0] Profilo esistente illeggibile, lo rigenero: {path}")
        return None
    if payload.get("detection", {}).get("signature") != signature:
        print("[f0] Profilo esistente non piu' valido (input o modello cambiati): lo rigenero.")
        return None
    try:
        validate_payload(payload, where=str(path))
    except LanguageProfileError as exc:
        print(f"[f0] Profilo esistente non valido ({exc}): lo rigenero.")
        return None
    print(f"[cache] Profilo linguistico riusato: {path}")
    return payload


# --------------------------------------------------------------------------- #
# Step                                                                         #
# --------------------------------------------------------------------------- #
def detect_language(config: F1Config | None = None) -> dict:
    config = config or load_config()
    target = active_config_path(config.output_dir)

    blocks = load_blocks(config)
    sample = build_language_sample(blocks)
    signature = sample_signature(config, sample)
    print(f"[f0] Campione per il riconoscimento: {len(sample)} caratteri da {len(blocks)} blocchi.")

    cached = load_cached_profile(target, signature)
    if cached is not None:
        return cached

    hint = deterministic_language_hint(sample)
    if FORCED_LANGUAGE:
        detection = {
            "code": FORCED_LANGUAGE,
            "name": FORCED_LANGUAGE,
            "english_name": FORCED_LANGUAGE,
            "confidence": 1.0,
            "evidence": "F0_FORCE_LANGUAGE",
        }
        print(f"[f0] Lingua imposta da F0_FORCE_LANGUAGE: {FORCED_LANGUAGE}")
    else:
        detection = detect_language_with_llm(sample, config)
        print(
            f"[f0] Lingua riconosciuta: {detection['code']} ({detection['name']}), "
            f"confidenza {detection['confidence']}."
        )

    detection["deterministic_hint"] = hint
    # La controprova non decide, ma va registrata: se l'LLM dice 'en' mentre le
    # stopword italiane sono fittissime, il report lo mostra invece di lasciare
    # il disallineamento sepolto in un albero vuoto tre step piu' avanti.
    if hint["best"] and hint["best"] != detection["code"] and (hint["best_score"] or 0) >= 0.05:
        print(
            f"[f0] ATTENZIONE: l'LLM indica '{detection['code']}' ma le stopword "
            f"'{hint['best']}' coprono il {hint['best_score']:.1%} del campione."
        )
        detection["hint_disagreement"] = True

    baseline = bundled_config_path(detection["code"])
    if baseline.exists() and not ALWAYS_GENERATE:
        # La baseline versionata e' il profilo tarato e testato per questa lingua:
        # rigenerarlo con un LLM sostituirebbe regex verificate con regex
        # plausibili. Usa F0_ALWAYS_GENERATE=1 per forzare la rigenerazione.
        payload = json.loads(baseline.read_text(encoding="utf-8"))
        payload["source"] = f"baseline:{baseline.name}"
        coverage = structural_coverage(payload, sample)
        print(f"[f0] Baseline versionata trovata: {baseline.name}. Copertura intestazioni: {coverage}")
    else:
        if not baseline.exists():
            print(f"[f0] Nessuna baseline per '{detection['code']}': genero il profilo con l'LLM.")
        payload, coverage = generate_profile_with_llm(sample, detection, config)

    payload["detection"] = {
        **detection,
        "signature": signature,
        "structural_coverage": coverage,
        "document": str(config.blocks_json_path),
        "baselines_available": bundled_language_codes(),
    }

    validate_payload(payload, where=str(target))
    write_json(target, payload)
    print(f"[OK] Profilo linguistico scritto: {target}")
    print(f"[f0] Lingua attiva per gli step successivi: {payload['code']} ({payload['name']}).")
    return payload


def main() -> None:
    detect_language()


if __name__ == "__main__":
    main()
