"""Profilo linguistico della pipeline: fonte unica di regex e termini dipendenti dalla lingua.

Le parole chiave strutturali ("PARTE", "CAPO", "Art.", "considerando quanto
segue", "e' inserito il capo seguente"...) non sono scritte nei moduli delle
fasi: cambiare lingua significa cambiare profilo, non codice.

Qui vive UN solo oggetto, ``LanguageProfile``, costruito da un file JSON
(``config/language_<codice>.json`` come baseline versionata, oppure
``<OUTPUT_DIR>/language_config.json`` prodotto dallo step f0 per il documento
corrente). Il profilo espande i segnaposto dei pattern e compila le regex: gli
step consumano solo l'oggetto, mai i letterali.

Segnaposto ammessi nei pattern del file di configurazione (vedi
``_expand_placeholders``): ``{num}``, ``{article_num}``, ``{part_number}``,
``{suffix}``, ``{suffix_hyphenated}``, ``{opt_suffix}``, ``{dash}``,
``{quotes}``, ``{rubric_max}``. Le graffe che non corrispondono a un segnaposto
noto restano letterali, cosi' i quantificatori regex (``{1,4}``) si scrivono in
modo naturale.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PACKAGE_DIR.parent / "config"

SCHEMA_VERSION = 1
DEFAULT_LANGUAGE_CODE = "it"
ACTIVE_CONFIG_NAME = "language_config.json"

# Neutro rispetto alla lingua: trattini tipografici e livelli Akoma Ntoso non
# cambiano se il documento e' francese o inglese.
DASH_PATTERN = r"[-\u2010-\u2015]"

AKN_LEVELS: dict[str, int] = {
    "part": 1,
    "annex": 1,
    "recital": 1,
    "title": 2,
    "chapter": 3,
    "section": 4,
    "article": 5,
}
CONTENT_LABELS = {"paragraph", "point", "content", "unresolved"}
NOISE_LABELS = {"noise"}
ALLOWED_LABELS = set(AKN_LEVELS) | CONTENT_LABELS | NOISE_LABELS

# Etichette per cui il profilo deve fornire un pattern di intestazione: senza
# uno di questi la pipeline non sa piu' riconoscere quel livello.
REQUIRED_HEADER_LABELS = ("part", "title", "chapter", "section", "annex", "article", "recital")
REQUIRED_PATTERNS = (
    "article_components",
    "structural_marker",
    "modificative_instruction",
    "editorial_structural_note",
    "paragraph_marker",
    "point_marker",
    "page_number_noise",
    "recital_item",
    "recital_number",
    "recital_path_segment",
    "recital_section_start",
    "device_start",
    "preamble_marker",
    "annex_local_part",
    "footnote_citation_markers",
)
REQUIRED_DISPLAY_KEYS = ("recital_prefix", "path_prefix")

_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


class LanguageProfileError(ValueError):
    """Configurazione linguistica assente, incompleta o non compilabile."""


def _dal_piu_lungo(testo: str) -> tuple[int, str]:
    """Chiave d'ordinamento: prima i piu' lunghi, poi in ordine alfabetico."""
    return (-len(testo), testo)


def _expand_placeholders(template: str, mapping: dict[str, str], *, where: str) -> str:
    """Sostituisce i soli segnaposto noti, lasciando letterali le altre graffe.

    ``{suffix}`` viene sostituito; ``{1,4}`` (quantificatore regex) no. Questo
    evita di dover raddoppiare le graffe nel file di configurazione, cosa che
    l'LLM di f0 sbaglierebbe sistematicamente.
    """
    if not isinstance(template, str):
        raise LanguageProfileError(f"{where}: atteso un pattern testuale, trovato {type(template).__name__}")

    def sostituisci(trovato: re.Match) -> str:
        nome_segnaposto = trovato.group(1)
        # Se il nome non e' fra i segnaposto noti si restituisce il testo
        # originale, graffe comprese: e' un quantificatore regex, non un
        # segnaposto.
        return mapping.get(nome_segnaposto, trovato.group(0))

    return _PLACEHOLDER_RE.sub(sostituisci, template)


def _compile(pattern: str, *, where: str, flags: int = re.IGNORECASE) -> re.Pattern[str]:
    try:
        return re.compile(pattern, flags)
    except re.error as exc:
        raise LanguageProfileError(f"{where}: regex non compilabile ({exc}): {pattern}") from exc


def _require(payload: dict, key: str, expected: type, *, where: str):
    if key not in payload:
        raise LanguageProfileError(f"{where}: campo obbligatorio mancante '{key}'")
    value = payload[key]
    if not isinstance(value, expected):
        raise LanguageProfileError(
            f"{where}: '{key}' deve essere {expected.__name__}, trovato {type(value).__name__}"
        )
    return value


@dataclass
class LanguageProfile:
    """Tutto cio' che nella pipeline dipende dalla lingua del documento."""

    code: str
    name: str
    english_name: str
    quotes: str
    article_rubric_max_chars: int
    numbering: dict[str, Any]
    ordinal_suffixes: tuple[str, ...]
    suffix_aliases: dict[str, str]
    akn_patterns: list[dict[str, Any]]
    akn_document_types: list[dict[str, Any]]
    structure_conventions: dict[str, Any]
    header_patterns_source: dict[str, str]
    patterns_source: dict[str, str]
    label_descriptions: dict[str, str]
    label_examples: dict[str, list[str]]
    prompt_terms: dict[str, Any]
    display: dict[str, str]
    smoke_queries: dict[str, str] = field(default_factory=dict)
    detection_hints: dict[str, Any] = field(default_factory=dict)
    origin: str = "baseline"
    source_path: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    # --- derivati (costruiti in __post_init__) ------------------------------- #
    suffix_alternation: str = field(init=False)
    suffix_hyphenated_alternation: str = field(init=False)
    optional_suffix_pattern: str = field(init=False)
    roman_or_arabic_pattern: str = field(init=False)
    article_number_pattern: str = field(init=False)
    part_number_pattern: str = field(init=False)
    header_patterns: dict[str, re.Pattern[str]] = field(init=False)
    patterns: dict[str, re.Pattern[str]] = field(init=False)
    document_type_patterns: list[tuple[re.Pattern[str], dict[str, Any]]] = field(init=False)

    def __post_init__(self) -> None:
        # Suffissi ordinali: ripuliti, senza duplicati e ordinati dal piu' lungo
        # al piu' corto. L'ordine conta: nell'alternation di una regex,
        # "quinquiesdecies" deve essere provato prima di "quinquies", altrimenti
        # il piu' corto vince e la coda resta fuori dal numero.
        suffissi_puliti = set()
        for suffisso in self.ordinal_suffixes:
            if not suffisso or not suffisso.strip():
                continue
            suffissi_puliti.add(suffisso.strip().lower())
        self.ordinal_suffixes = tuple(sorted(suffissi_puliti, key=_dal_piu_lungo))

        alias_minuscoli = {}
        for chiave, valore in self.suffix_aliases.items():
            alias_minuscoli[chiave.lower()] = valore.lower()
        self.suffix_aliases = alias_minuscoli

        self.roman_or_arabic_pattern = self.numbering.get("roman_or_arabic") or r"\d+(?:\.\d+)?"
        self.article_number_pattern = self.numbering.get("article_number") or r"\d+(?:\.\d+)?"

        word_numerals = []
        for parola in self.numbering.get("word_numerals", []):
            if parola:
                word_numerals.append(parola)

        if self.ordinal_suffixes:
            self.suffix_alternation = "|".join(self.ordinal_suffixes)
            self.suffix_hyphenated_alternation = "|".join("-?".join(s) for s in self.ordinal_suffixes)
            self.optional_suffix_pattern = (
                rf"(?:\s*{DASH_PATTERN}?\s*"
                rf"((?:{self.suffix_alternation})(?:[\s-]+(?:{self.suffix_alternation}))?))?"
            )
        else:
            # Lingue senza suffissi ordinali: il gruppo di cattura deve esistere
            # comunque (extract_structural_number legge sempre group(2)) ma non
            # deve mai matchare, altrimenti divora il testo che segue il numero.
            self.suffix_alternation = "(?!)"
            self.suffix_hyphenated_alternation = "(?!)"
            self.optional_suffix_pattern = r"(?:(?!)())?"

        if word_numerals:
            self.part_number_pattern = f"(?:{'|'.join(word_numerals)}|{self.roman_or_arabic_pattern})"
        else:
            self.part_number_pattern = self.roman_or_arabic_pattern

        mapping = {
            "num": self.roman_or_arabic_pattern,
            "article_num": self.article_number_pattern,
            "part_number": self.part_number_pattern,
            "suffix": self.suffix_alternation,
            "suffix_hyphenated": self.suffix_hyphenated_alternation,
            "opt_suffix": self.optional_suffix_pattern,
            "dash": DASH_PATTERN,
            "quotes": re.escape(self.quotes),
            "rubric_max": str(self.article_rubric_max_chars),
        }
        self._placeholders = mapping

        self.akn_patterns = [
            {
                "etichetta": item["etichetta"],
                "pattern": _expand_placeholders(
                    item["pattern"], mapping, where=f"akn_patterns[{item.get('etichetta')}]"
                ),
                "priorita": item["priorita"],
            }
            for item in self.akn_patterns
        ]

        self.header_patterns = {
            label: _compile(
                _expand_placeholders(pattern, mapping, where=f"header_patterns[{label}]"),
                where=f"header_patterns[{label}]",
            )
            for label, pattern in self.header_patterns_source.items()
        }
        self.patterns = {
            key: _compile(
                _expand_placeholders(pattern, mapping, where=f"patterns[{key}]"),
                where=f"patterns[{key}]",
            )
            for key, pattern in self.patterns_source.items()
        }
        self.document_type_patterns = [
            (
                _compile(
                    _expand_placeholders(rule["pattern"], mapping, where=f"akn_document_types[{index}]"),
                    where=f"akn_document_types[{index}]",
                ),
                rule,
            )
            for index, rule in enumerate(self.akn_document_types)
        ]

    # Convenzioni tipografiche di redazione, con i default che riproducono il
    # comportamento storico (italiano). Erano costanti dentro f2: la posizione
    # della rubrica rispetto al numero e' una proprieta' della tecnica
    # legislativa, non del codice — in italiano numero e rubrica stanno sulla
    # stessa riga ("Art. 5 Definizioni"), nel common law la rubrica e' una riga
    # separata SOPRA il testo numerato.
    STRUCTURE_DEFAULTS: ClassVar[dict[str, Any]] = {
        "article_rubric_position": "after",
        "article_rubric_max_words": 30,
        "rubric_block_types": (),
        "rubric_lookahead_blocks": 3,
        "contents_page_min_headings": 5,
        "contents_page_min_ratio": 0.5,
        "running_header_max_page_order": 2,
        "running_header_min_pages": 3,
        "recital_sequence_tolerance": 3,
        # Marcatore testuale di un blocco scritto in una lingua DIVERSA da
        # quella del documento (es. una fonte ufficialmente bilingue che
        # pubblica ogni articolo due volte). Nessun default universale ha
        # senso: e' None finche' il profilo di una lingua non lo dichiara, e
        # in quel caso il controllo e' semplicemente disattivo (nessun blocco
        # viene mai scartato per questo motivo).
        "foreign_text_marker_pattern": None,
        # Quota minima di parole del blocco che devono contenere il marcatore
        # perche' sia trattato come rumore multilingue, non un prestito
        # occasionale.
        "foreign_text_min_marker_ratio": 0.03,
        # Sotto questa lunghezza il rapporto e' troppo rumoroso per decidere:
        # un titolo di due parole non basta a stabilire la lingua.
        "foreign_text_min_words": 6,
    }

    def convention(self, key: str) -> Any:
        """Valore di una convenzione strutturale, con ripiego sul default.

        Il ripiego e' su un default UNICO e dichiarato, non sul valore italiano
        travestito da universale: un profilo che non dichiara nulla si comporta
        come si comportava la pipeline prima che la sezione esistesse.
        """
        if key not in self.STRUCTURE_DEFAULTS:
            raise KeyError(f"convenzione strutturale non prevista: {key!r}")
        value = (self.structure_conventions or {}).get(key)
        return self.STRUCTURE_DEFAULTS[key] if value is None else value

    def document_type_rules(self) -> list[tuple[re.Pattern[str], dict[str, Any]]]:
        """Regole per riconoscere il tipo di atto dal suo incipit.

        Ripiego sulla baseline versionata della STESSA lingua quando il profilo
        attivo non definisce la sezione: ``akn_document_types`` e' arrivata dopo
        i primi profili scritti da f0, e quelli non la contengono. Rigenerarli
        costerebbe una rilettura del documento con l'LLM per un'informazione che
        e' proprieta' della lingua, non del singolo documento.
        """
        if self.document_type_patterns or self.origin == "baseline":
            return self.document_type_patterns
        if getattr(self, "_fallback_document_types", None) is None:
            baseline = bundled_config_path(self.code)
            try:
                self._fallback_document_types = (
                    read_profile_file(baseline, origin="baseline").document_type_patterns
                    if baseline.exists()
                    else []
                )
            except LanguageProfileError:
                self._fallback_document_types = []
        return self._fallback_document_types

    # --- accessori usati dai prompt ------------------------------------------ #
    def examples(self, label: str, limit: int | None = None) -> list[str]:
        items = [str(x) for x in self.label_examples.get(label, []) if str(x).strip()]
        return items[:limit] if limit else items

    def examples_text(self, label: str, *, limit: int | None = None, sep: str = ", ") -> str:
        return sep.join(self.examples(label, limit))

    def label_line(self, label: str) -> str:
        return f"- {label}: {self.label_descriptions.get(label, label)}"

    def allowed_labels_text(self) -> str:
        return "\n".join(self.label_line(label) for label in sorted(ALLOWED_LABELS))

    def hierarchy_glosses_text(
        self, *, indent: str = "   ", suffix: str = ";", exclude: tuple[str, ...] = ()
    ) -> str:
        glosses = self.prompt_terms.get("hierarchy_glosses", {})
        return "\n".join(
            f"{indent}- {label}: {gloss}{suffix}"
            for label, gloss in glosses.items()
            if label not in exclude
        )

    def forbidden_synonyms_text(self) -> str:
        return ", ".join(self.prompt_terms.get("forbidden_label_synonyms", []))

    def modificative_examples_text(self) -> str:
        return ", ".join(f'"{x}"' for x in self.prompt_terms.get("modificative_examples", []))

    def article_citation_examples_text(self) -> str:
        return " oppure ".join(self.prompt_terms.get("article_citation_examples", []))

    @property
    def recital_prefix(self) -> str:
        return self.display.get("recital_prefix", "Recital")

    @property
    def path_prefix(self) -> str:
        return self.display.get("path_prefix", "Path")

    def smoke_query(self, key: str, default: str) -> str:
        """Query di prova degli step diagnostici (ricerca nel grafo, top-k vettoriale).

        Non incidono sugli artefatti della pipeline, ma una query italiana su un
        corpus inglese restituisce zero risultati e fa sembrare rotto il caricamento.
        """
        return self.smoke_queries.get(key) or default

    def fingerprint(self) -> dict[str, Any]:
        """Identita' del profilo, per le firme di cache degli step."""
        return {
            "code": self.code,
            "origin": self.origin,
            "source_path": str(self.source_path) if self.source_path else None,
            "sha256": _payload_digest(self.raw),
        }


# Chiavi escluse dall'impronta: commenti e metadati di riconoscimento non
# cambiano il comportamento della pipeline. Includerle farebbe invalidare la
# cache di f1a a ogni riesecuzione di f0, anche a profilo identico.
_DIGEST_IGNORED_KEYS = {"detection", "source"}


def payload_digest(payload: dict[str, Any]) -> str:
    """Impronta del contenuto di un profilo, al netto di commenti e metadati."""
    import hashlib

    serializable = {
        k: v for k, v in payload.items() if not k.startswith("_") and k not in _DIGEST_IGNORED_KEYS
    }
    return hashlib.sha256(json.dumps(serializable, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def baseline_digest(code: str = DEFAULT_LANGUAGE_CODE) -> str:
    """Impronta della baseline versionata di una lingua."""
    return payload_digest(json.loads(bundled_config_path(code).read_text(encoding="utf-8")))


_payload_digest = payload_digest  # nome storico interno


# --------------------------------------------------------------------------- #
# Caricamento                                                                  #
# --------------------------------------------------------------------------- #
def validate_payload(payload: dict[str, Any], *, where: str = "profilo") -> None:
    """Controlla struttura e compilabilita' senza costruire il profilo.

    Usata da f0 per rifiutare un profilo generato dall'LLM prima di scriverlo su
    disco: un profilo scritto e non compilabile bloccherebbe tutti gli step
    successivi con un errore lontano dalla causa.
    """
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise LanguageProfileError(
            f"{where}: schema_version {payload.get('schema_version')!r} non supportata (attesa {SCHEMA_VERSION})"
        )
    for key in ("code", "name", "quotes"):
        _require(payload, key, str, where=where)
    _require(payload, "numbering", dict, where=where)
    _require(payload, "ordinal_suffixes", list, where=where)
    _require(payload, "suffix_aliases", dict, where=where)
    _require(payload, "akn_patterns", list, where=where)
    header_patterns = _require(payload, "header_patterns", dict, where=where)
    patterns = _require(payload, "patterns", dict, where=where)
    label_descriptions = _require(payload, "label_descriptions", dict, where=where)
    display = _require(payload, "display", dict, where=where)

    missing_headers = [label for label in REQUIRED_HEADER_LABELS if not header_patterns.get(label)]
    if missing_headers:
        raise LanguageProfileError(f"{where}: header_patterns mancanti per {missing_headers}")

    missing_patterns = [key for key in REQUIRED_PATTERNS if not patterns.get(key)]
    if missing_patterns:
        raise LanguageProfileError(f"{where}: patterns mancanti per {missing_patterns}")

    missing_labels = sorted(ALLOWED_LABELS - set(label_descriptions))
    if missing_labels:
        raise LanguageProfileError(f"{where}: label_descriptions mancanti per {missing_labels}")

    unknown_labels = sorted(set(label_descriptions) - ALLOWED_LABELS)
    if unknown_labels:
        raise LanguageProfileError(f"{where}: label non ammesse in label_descriptions: {unknown_labels}")

    akn_labels = {item.get("etichetta") for item in payload["akn_patterns"] if isinstance(item, dict)}
    unknown_akn = sorted(akn_labels - ALLOWED_LABELS)
    if unknown_akn:
        raise LanguageProfileError(f"{where}: label non ammesse in akn_patterns: {unknown_akn}")

    missing_display = [key for key in REQUIRED_DISPLAY_KEYS if not display.get(key)]
    if missing_display:
        raise LanguageProfileError(f"{where}: display mancante per {missing_display}")

    # Sezione facoltativa: i profili generati da f0 prima della sua introduzione
    # restano validi e ripiegano sulla baseline (vedi document_type_rules).
    for index, rule in enumerate(payload.get("akn_document_types", [])):
        if not isinstance(rule, dict) or not rule.get("pattern"):
            raise LanguageProfileError(f"{where}: akn_document_types[{index}] senza campo 'pattern'")

    # Compilazione di prova: fallisce qui, non a meta' pipeline.
    build_profile(payload, origin="validation")


def build_profile(payload: dict[str, Any], *, origin: str = "baseline", source_path: Path | None = None) -> LanguageProfile:
    return LanguageProfile(
        code=payload["code"],
        name=payload.get("name", payload["code"]),
        english_name=payload.get("english_name", payload.get("name", payload["code"])),
        quotes=payload["quotes"],
        article_rubric_max_chars=int(payload.get("article_rubric_max_chars", 160)),
        numbering=dict(payload.get("numbering", {})),
        ordinal_suffixes=tuple(payload.get("ordinal_suffixes", [])),
        suffix_aliases=dict(payload.get("suffix_aliases", {})),
        akn_patterns=[dict(item) for item in payload.get("akn_patterns", [])],
        akn_document_types=[dict(item) for item in payload.get("akn_document_types", [])],
        structure_conventions=dict(payload.get("structure_conventions", {})),
        header_patterns_source=dict(payload.get("header_patterns", {})),
        patterns_source=dict(payload.get("patterns", {})),
        label_descriptions=dict(payload.get("label_descriptions", {})),
        label_examples={k: list(v) for k, v in payload.get("label_examples", {}).items()},
        prompt_terms=dict(payload.get("prompt_terms", {})),
        display=dict(payload.get("display", {})),
        smoke_queries=dict(payload.get("smoke_queries", {})),
        detection_hints=dict(payload.get("detection_hints", {})),
        origin=origin,
        source_path=source_path,
        raw=payload,
    )


def read_profile_file(path: Path, *, origin: str) -> LanguageProfile:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LanguageProfileError(f"Profilo linguistico illeggibile: {path} ({exc})") from exc
    validate_payload(payload, where=str(path))
    return build_profile(payload, origin=origin, source_path=path)


def bundled_config_path(code: str) -> Path:
    return CONFIG_DIR / f"language_{code.lower()}.json"


def bundled_language_codes() -> list[str]:
    return sorted(p.stem.removeprefix("language_") for p in CONFIG_DIR.glob("language_*.json"))


def active_config_path(output_dir: str | Path | None = None) -> Path:
    """Percorso del profilo generato da f0 per il documento corrente."""
    explicit = os.getenv("LANGUAGE_CONFIG_PATH")
    if explicit:
        return Path(explicit)
    base = Path(output_dir) if output_dir is not None else Path(os.getenv("OUTPUT_DIR", "."))
    return base / ACTIVE_CONFIG_NAME


def load_language_profile(output_dir: str | Path | None = None) -> LanguageProfile:
    """Profilo attivo: quello scritto da f0 se esiste, altrimenti la baseline.

    L'ordine e' deliberato. Il profilo del documento (prodotto da f0 nella sua
    cartella di output) vince sempre, cosi' documenti in lingue diverse nello
    stesso batch usano regex diverse. Se f0 non e' stato eseguito si ricade sulla
    baseline versionata indicata da ``LANGUAGE_CODE`` (default italiano), che e'
    il comportamento storico della pipeline.
    """
    path = active_config_path(output_dir)
    if path.exists():
        return read_profile_file(path, origin="f0")

    code = os.getenv("LANGUAGE_CODE", DEFAULT_LANGUAGE_CODE)
    baseline = bundled_config_path(code)
    if not baseline.exists():
        raise LanguageProfileError(
            f"Nessun profilo linguistico per '{code}': manca {baseline} e non e' stato eseguito lo step f0. "
            f"Baseline disponibili: {bundled_language_codes()}"
        )
    return read_profile_file(baseline, origin="baseline")


_cached_profile: LanguageProfile | None = None


def active_profile() -> LanguageProfile:
    """Profilo attivo del processo corrente (memoizzato).

    Ogni step gira in un processo separato con il proprio ``OUTPUT_DIR``: la
    memoizzazione a livello di processo non puo' far trapelare il profilo di un
    documento in quello successivo.
    """
    global _cached_profile
    if _cached_profile is None:
        _cached_profile = load_language_profile()
    return _cached_profile


def reset_active_profile() -> None:
    """Invalida la memoizzazione (usata dai test e da f0 dopo la scrittura)."""
    global _cached_profile
    _cached_profile = None
