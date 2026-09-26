from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Tutto cio' che dipende dalla lingua del documento (parole chiave strutturali,
# regex di intestazione, descrizioni delle etichette) arriva dal profilo attivo:
# la baseline versionata ``config/language_it.json`` oppure, quando lo step f0 e'
# stato eseguito, ``<OUTPUT_DIR>/language_config.json`` scritto per QUESTO
# documento. I nomi esposti qui sotto restano quelli storici del modulo, e sono
# solo quelli che gli step consumano davvero: il profilo completo resta
# raggiungibile come ``LANGUAGE.<campo>``, quindi un alias senza lettori sarebbe
# un secondo nome da tenere allineato a mano senza che nessuno lo usi.
#
# NB: normalize_number (sotto) resta VOLUTAMENTE distinta dalla forma canonica
# usata dai validatori esterni per confrontare i numeri: la pipeline
# preserva i numeri romani ("Capo IV" -> "IV") in tutti gli output prodotti.
from legisleaf.language import (
    # AKN_LEVELS e CONTENT_LABELS non sono usati QUI: sono riesportati, perche'
    # f1a e f2_build_tree li importano da questo modulo insieme a tutto il resto
    # del vocabolario. Non rimuoverli perche' un analizzatore li segnala come
    # inutilizzati nel file: guardare prima chi importa da f1_common.
    AKN_LEVELS,  # noqa: F401  (re-export per f1a, f2_build_tree)
    ALLOWED_LABELS,
    CONTENT_LABELS,  # noqa: F401  (re-export per f2_build_tree)
    DASH_PATTERN,
    NOISE_LABELS,
    active_profile,
)

# Default dei modelli e dell'endpoint: unica fonte in legisleaf.settings, cosi' un
# cambio di modello non richiede di ricordarsi di questo file.
from legisleaf.settings import (
        LLM_MODEL_NAME as DEFAULT_LLM_MODEL_NAME,
        OPENAI_API_KEY as DEFAULT_OPENAI_API_KEY,
        OPENAI_BASE_URL as DEFAULT_OPENAI_BASE_URL,
        SLM_MODEL_NAME as DEFAULT_SLM_MODEL_NAME,
    )

LANGUAGE = active_profile()

LATIN_SUFFIX_PATTERN = LANGUAGE.suffix_alternation
CANONICAL_SUFFIX_ALIASES = LANGUAGE.suffix_aliases
ROMAN_OR_ARABIC_PATTERN = LANGUAGE.roman_or_arabic_pattern
LEGISLATIVE_QUOTES = LANGUAGE.quotes
ARTICLE_RUBRIC_MAX_CHARS = LANGUAGE.article_rubric_max_chars

AKN_PATTERNS = LANGUAGE.akn_patterns
LABEL_DESCRIPTIONS = LANGUAGE.label_descriptions

HEADER_PATTERNS = LANGUAGE.header_patterns
ARTICLE_HEADER_RE = HEADER_PATTERNS["article"]
ANNEX_HEADER_RE = HEADER_PATTERNS["annex"]
RECITAL_HEADER_RE = HEADER_PATTERNS["recital"]
ARTICLE_HEADER_COMPONENT_RE = LANGUAGE.patterns["article_components"]
STRUCTURAL_MARKER_RE = LANGUAGE.patterns["structural_marker"]
MODIFICATIVE_INSTRUCTION_RE = LANGUAGE.patterns["modificative_instruction"]
EDITORIAL_STRUCTURAL_NOTE_RE = LANGUAGE.patterns["editorial_structural_note"]
PARAGRAPH_MARKER_RE = LANGUAGE.patterns["paragraph_marker"]
POINT_MARKER_RE = LANGUAGE.patterns["point_marker"]
PAGE_NUMBER_NOISE_RE = LANGUAGE.patterns["page_number_noise"]
RECITAL_NUMBER_RE = LANGUAGE.patterns["recital_number"]


def normalize_source_text(text: str | None) -> str:
    """Normalize whitespace and dashes while preserving semantically relevant glyphs."""
    normalized = (text or "").replace("\xa0", " ")
    normalized = re.sub(r"[\u2010-\u2015]", "-", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"\s+([,.;:])", r"\1", normalized)
    normalized = re.sub(r"[.;:,]+$", "", normalized).strip()
    return normalized


def normalize_matching_text(text: str | None) -> str:
    """Normalize text for matching and dedupe keys."""
    normalized = normalize_source_text(text)
    return normalized.translate({ord(ch): "" for ch in LEGISLATIVE_QUOTES})


def normalize_legal_text(text: str | None) -> str:
    """Backward-compatible matching normalization."""
    return normalize_matching_text(text)


def normalize_suffix(suffix: str | None) -> str | None:
    if not suffix:
        return None
    value = re.sub(r"[\s-]+", "", suffix.lower().strip())
    return CANONICAL_SUFFIX_ALIASES.get(value, value) if value else None


def normalize_number(number: str | None) -> str | None:
    if not number:
        return None
    value = normalize_legal_text(number).strip(" .;:()[]")
    if not value:
        return None
    value = re.sub(rf"\s*{DASH_PATTERN}\s*", "-", value)
    match = re.match(
        rf"^({ROMAN_OR_ARABIC_PATTERN})(?:[-\s]+((?:{LATIN_SUFFIX_PATTERN})(?:[-\s]+(?:{LATIN_SUFFIX_PATTERN}))?))?\b",
        value,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    base = match.group(1)
    suffix = normalize_suffix(match.group(2))
    base = base.upper() if re.fullmatch(r"[ivxlcdm]+(?:\.\d+)?", base, flags=re.IGNORECASE) else base.lower()
    return f"{base}-{suffix}" if suffix else base


def is_modificative_instruction(text: str | None) -> bool:
    return bool(MODIFICATIVE_INSTRUCTION_RE.search(normalize_legal_text(text)))


def is_structural_editorial_note(text: str | None) -> bool:
    normalized = normalize_legal_text(text)
    return bool(EDITORIAL_STRUCTURAL_NOTE_RE.match(normalized) or is_modificative_instruction(normalized))


def is_article_header(text: str | None) -> bool:
    normalized = normalize_legal_text(text)
    if not normalized or is_structural_editorial_note(normalized):
        return False
    match = ARTICLE_HEADER_RE.match(normalized)
    if not match:
        return False
    rubric = (match.groupdict().get("rubrica") or "").strip()
    return len(rubric) <= ARTICLE_RUBRIC_MAX_CHARS


def is_annex_header(text: str | None) -> bool:
    normalized = normalize_legal_text(text)
    return bool(normalized and ANNEX_HEADER_RE.match(normalized) and not is_structural_editorial_note(normalized))


def is_recital_header(text: str | None) -> bool:
    normalized = normalize_legal_text(text)
    return bool(normalized and RECITAL_HEADER_RE.match(normalized) and not is_structural_editorial_note(normalized))


def is_strict_header(text: str | None, node_label: str) -> bool:
    normalized = normalize_legal_text(text)
    if not normalized or is_structural_editorial_note(normalized):
        return False
    pattern = HEADER_PATTERNS.get(node_label)
    return bool(pattern and pattern.match(normalized))


def strict_label_for_text(text: str | None, predicted_label: str | None) -> str | None:
    label = (predicted_label or "").lower().strip()
    if label == "article":
        return "article" if is_article_header(text) else None
    if label == "annex":
        return "annex" if is_annex_header(text) else None
    if label == "recital":
        return "recital" if is_recital_header(text) else None
    if label in {"part", "title", "chapter", "section"}:
        return label if is_strict_header(text, label) else None
    if label in CONTENT_LABELS | NOISE_LABELS:
        return label
    return None


STRUCTURAL_LABELS = {"part", "title", "chapter", "section", "annex", "recital", "article"}


def apply_strict_structural_validation(label: str, text: str, source: dict) -> tuple[str, dict | None]:
    """Nessun modello puo' promuovere un blocco a nodo strutturale se il testo non
    ne ha la forma.

    Un'etichetta strutturale crea un NODO nell'albero: se il testo non e'
    un'intestazione, quel nodo e' un fantasma senza numero che frammenta la
    gerarchia. La rubrica di un articolo ("Definizioni", "Sanzioni", "Oggetto") e'
    l'errore tipico: e' l'unica riga breve subito sotto l'intestazione, e un
    classificatore la scambia volentieri per l'intestazione stessa.

    Il controllo e' puramente testuale e va applicato a OGNI sorgente di etichette
    (SLM in f1b e revisione LLM in f1c): applicarlo a una sola delle due lascia
    aperta esattamente la strada che si voleva chiudere.

    Ritorna ``(etichetta_finale, motivo_declassamento | None)``.
    """
    validated = strict_label_for_text(text, label)
    if validated:
        return validated, None
    if label in STRUCTURAL_LABELS:
        reason = {
            "from": label,
            "to": "unresolved" if source.get("recovered_extra_candidate") else "content",
            "reason": "strict_structural_validation_failed",
        }
        return reason["to"], reason
    return label, None


def infer_structural_label(text: str | None) -> str | None:
    normalized = normalize_legal_text(text)
    if not normalized or is_structural_editorial_note(normalized):
        return None
    for label in ("annex", "article", "part", "title", "chapter", "section", "recital"):
        if HEADER_PATTERNS[label].match(normalized):
            return label
    return None


def extract_structural_number(text: str | None, node_label: str) -> str | None:
    normalized = normalize_legal_text(text)
    if not normalized:
        return None
    if node_label == "article":
        match = ARTICLE_HEADER_COMPONENT_RE.match(normalized)
        if not match:
            return None
        return normalize_number(" ".join(part for part in [match.group("number"), match.group("suffix")] if part))
    if node_label == "recital":
        match = RECITAL_NUMBER_RE.match(normalized)
        if not match:
            return None
        return match.group(1) or match.group(2)
    pattern = HEADER_PATTERNS.get(node_label)
    if pattern is None:
        return None
    match = pattern.match(normalized)
    if not match:
        return None
    base = match.group(1)
    suffix = match.group(2) if len(match.groups()) >= 2 else None
    return normalize_number(" ".join(part for part in [base, suffix] if part))


def tronca(testo: str, max_chars: int) -> str:
    testo = (testo or "").strip()
    if len(testo) <= max_chars:
        return testo
    return testo[:max_chars] + " [...]"


def split_article_heading(text: str | None) -> tuple[str, str | None]:
    original = (text or "").strip()
    normalized = normalize_legal_text(original)
    match = ARTICLE_HEADER_COMPONENT_RE.match(normalized)
    if not match:
        return original, None
    number = normalize_number(" ".join(part for part in [match.group("number"), match.group("suffix")] if part))
    heading = f"{match.group('prefix')} {number}" if number else original
    rubric = (match.group("rubrica") or "").strip(" .;:-")
    return heading, rubric or None


@dataclass(frozen=True)
class F1Config:
    output_dir: Path
    blocks_json_path: Path
    analysis_json_path: Path
    embedding_db_path: Path
    classification_results_path: Path
    classification_slm_raw_path: Path
    low_confidence_results_path: Path
    llm_reviewed_results_path: Path
    classification_failures_path: Path
    llm_review_raw_errors_path: Path
    model_name: str
    model_name_local: str
    openai_base_url: str
    openai_api_key: str
    n_sample_pages: int
    max_llm_sample_chars: int
    n_llm_examples: int
    n_similar_examples: int
    max_chars_testo: int
    max_retries: int
    llm_timeout_seconds: int
    use_cached_analysis: bool
    enable_logprobs: bool
    confidence_accept_threshold: float
    process_low_conf_with_llm: bool
    llm_review_batch_size: int
    llm_review_reasoning_effort: str
    llm_review_max_workers: int
    n_file: int | None
    embedding_model_name: str
    embedding_device: str


def env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default) == "1"


def load_config() -> F1Config:
    output_dir = Path(os.getenv("OUTPUT_DIR", "."))
    blocks_json_path = Path(os.getenv("BLOCKS_JSON_PATH", "codice_strada_major_voting.json"))
    n_file = os.getenv("N_FILE")
    output_dir.mkdir(parents=True, exist_ok=True)
    return F1Config(
        output_dir=output_dir,
        blocks_json_path=blocks_json_path,
        analysis_json_path=output_dir / os.getenv("ANALYSIS_JSON_NAME", "output_analysis.json"),
        embedding_db_path=output_dir / os.getenv("EMBEDDING_DB_NAME", "embedding_examples_db.npz"),
        classification_results_path=output_dir / os.getenv("CLASSIFICATION_RESULTS_NAME", "classificazione_blocchi.json"),
        classification_slm_raw_path=output_dir / os.getenv("CLASSIFICATION_SLM_RAW_NAME", "classificazione_blocchi_slm_raw.json"),
        low_confidence_results_path=output_dir / os.getenv("LOW_CONFIDENCE_RESULTS_NAME", "classificazione_bassa_confidenza.json"),
        llm_reviewed_results_path=output_dir / os.getenv("LLM_REVIEWED_RESULTS_NAME", "classificazione_bassa_confidenza_llm.json"),
        classification_failures_path=output_dir / os.getenv("CLASSIFICATION_FAILURES_NAME", "classificazione_fallimenti.json"),
        llm_review_raw_errors_path=output_dir / os.getenv("LLM_REVIEW_RAW_ERRORS_NAME", "llm_review_raw_errors.json"),
        # I default arrivano da legisleaf.settings, unica fonte: erano duplicati qui
        # come letterali, e al cambio di modello divergevano in silenzio. Eseguendo
        # uno step come modulo (senza legisleaf.run, che imposta le variabili
        # d'ambiente) si sarebbe interrogato il modello sbagliato.
        model_name=os.getenv("LLM_MODEL_NAME", DEFAULT_LLM_MODEL_NAME),
        model_name_local=os.getenv("SLM_MODEL_NAME", DEFAULT_SLM_MODEL_NAME),
        openai_base_url=os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
        openai_api_key=os.getenv("OPENAI_API_KEY", DEFAULT_OPENAI_API_KEY),
        n_sample_pages=int(os.getenv("N_SAMPLE_PAGES", "5")),
        max_llm_sample_chars=int(os.getenv("MAX_LLM_SAMPLE_CHARS", "12000")),
        n_llm_examples=int(os.getenv("N_LLM_EXAMPLES", "18")),
        n_similar_examples=int(os.getenv("N_SIMILAR_EXAMPLES", "3")),
        max_chars_testo=int(os.getenv("MAX_CHARS_TESTO", "900")),
        max_retries=int(os.getenv("MAX_RETRIES", "3")),
        llm_timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "900")),
        use_cached_analysis=env_bool("USE_CACHED_ANALYSIS", "1"),
        enable_logprobs=env_bool("ENABLE_LOGPROBS", "1"),
        # 0.93 era tarato quando la confidenza era sempre None e la soglia era
        # inerte: nessun blocco la superava, tutti passavano dalla revisione LLM.
        # Ora che i logprobs arrivano davvero, il valore va scelto sui dati. Sul
        # corpus inglese la revisione LLM ha confermato lo SLM nel 93.8% dei casi
        # (15942 blocchi rivisti, 983 modificati): la soglia serve a intercettare
        # quel 6%, non a rivedere tutto. 0.90 sulla probabilita' MEDIA PER TOKEN
        # (vedi label_confidence_from_logprobs) e' il compromesso di partenza; va
        # ricalibrato dopo la prima run con i logprobs attivi, confrontando la
        # confidenza registrata con i casi in cui l'LLM ha effettivamente
        # cambiato etichetta.
        confidence_accept_threshold=float(os.getenv("CONFIDENCE_ACCEPT_THRESHOLD", "0.90")),
        process_low_conf_with_llm=env_bool("PROCESS_LOW_CONF_WITH_LLM", "1"),
        llm_review_batch_size=int(os.getenv("LLM_REVIEW_BATCH_SIZE", "8")),
        llm_review_reasoning_effort=os.getenv("LLM_REVIEW_REASONING_EFFORT", "low"),
        llm_review_max_workers=int(os.getenv("LLM_REVIEW_MAX_WORKERS", "4")),
        n_file=int(n_file) if n_file else None,
        embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2"),
        embedding_device=os.getenv("EMBEDDING_DEVICE", "cuda"),
    )


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File richiesto mancante: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_blocks(config: F1Config) -> list[dict[str, Any]]:
    with config.blocks_json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    raw_blocks = data.get("blocks", data if isinstance(data, list) else [])
    blocks = []
    for i, block in enumerate(raw_blocks):
        testo = (block.get("text") or block.get("content") or block.get("testo") or "").strip()
        if not testo:
            continue
        blocks.append(
            {
                "file": config.blocks_json_path.name,
                "ordine": i,
                "page": block.get("page_index"),
                "page_order": block.get("page_reading_index", i),
                "block_type": block.get("block_type"),
                "block_type_original": block.get("block_type_original", block.get("original_block_type")),
                "block_type_canonical": block.get(
                    "block_type_canonical",
                    block.get("canonical_block_type", block.get("block_type")),
                ),
                "document_zone": block.get("document_zone", block.get("zone")),
                "bbox": block.get("bbox") or block.get("box_coordinate"),
                "bbox_inherited": bool(block.get("bbox_inherited")),
                "ocr_score": block.get("ocr_score", block.get("confidenza")),
                "recovered_extra_candidate": bool(block.get("recovered_extra_candidate")),
                "selected_model": block.get("selected_from", block.get("selected_model")),
                "source_model": block.get("source_model", block.get("model", block.get("selected_from"))),
                "source_order": block.get("source_order", block.get("page_reading_index", i)),
                "selected_from": block.get("selected_from"),
                "page_consensus_accuracy": block.get("page_consensus_accuracy"),
                "page_agreement_score": block.get("page_agreement_score", block.get("page_consensus_accuracy")),
                "testo": testo,
                "block": block,
            }
        )

    blocks.sort(key=lambda x: ((x["page"] is None), x["page"] or 0, x["page_order"]))
    if config.n_file is not None:
        blocks = blocks[: config.n_file]
    print("Blocchi letti:", len(blocks))
    print("File input:", config.blocks_json_path)
    return blocks


def allowed_text() -> str:
    return "\n".join(f"- {label}: {LABEL_DESCRIPTIONS[label]}" for label in sorted(ALLOWED_LABELS))


def valid_labels(items: list[Any], field: str) -> bool:
    labels = set()
    for item in items:
        if isinstance(item, dict):
            label = (item.get(field) or "").lower().strip()
            if label:
                labels.add(label)
        elif isinstance(item, str) and item.strip():
            labels.add(item.lower().strip())
    return labels <= ALLOWED_LABELS
