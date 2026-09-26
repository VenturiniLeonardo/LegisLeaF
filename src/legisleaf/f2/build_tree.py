from __future__ import annotations

import os
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from legisleaf.f2.akn import (
    akn_identity_from_document,
    akn_tree_to_xml,
    build_akn_tree,
    check_akn_tree,
    validate_against_xsd,
)
from legisleaf.common import (
    AKN_LEVELS,
    CONTENT_LABELS,
    LANGUAGE,
    NOISE_LABELS,
    PARAGRAPH_MARKER_RE,
    POINT_MARKER_RE,
    STRUCTURAL_MARKER_RE,
    extract_structural_number,
    infer_structural_label,
    is_article_header,
    is_structural_editorial_note,
    # Lettura e scrittura JSON: la stessa che usano f1a/f1b/f1c. Qui esistevano
    # due copie identiche (``read_json``/``write_json``) e sarebbero divergute
    # alla prima modifica di una delle due.
    load_json as read_json,
    normalize_legal_text,
    normalize_number,
    split_article_heading,
    write_json,
)


OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "."))
INPUT_JSON_PATH = Path(os.getenv("BLOCKS_JSON_PATH", os.getenv("OCR_JSON_PATH", "")))
DOCUMENT_ID = os.getenv("DOCUMENT_ID") or (INPUT_JSON_PATH.stem if INPUT_JSON_PATH.name else "document")
DOCUMENT_NAME = os.getenv("DOCUMENT_NAME") or (INPUT_JSON_PATH.stem if INPUT_JSON_PATH.name else "Documento")
ANALYSIS_JSON_PATH = OUTPUT_DIR / os.getenv("ANALYSIS_JSON_NAME", "output_analysis.json")
CLASSIFICATION_RESULTS_PATH = OUTPUT_DIR / os.getenv("CLASSIFICATION_RESULTS_NAME", "classificazione_blocchi.json")
FINAL_JSON_PATH = OUTPUT_DIR / os.getenv("FINAL_JSON_NAME", "struttura_ad_albero.json")
TREE_VALIDATION_REPORT_PATH = OUTPUT_DIR / os.getenv("TREE_VALIDATION_REPORT_NAME", "tree_validation_report.json")
RAG_JSON_PATH = OUTPUT_DIR / os.getenv("RAG_JSON_NAME", "struttura_rag_graph.json")
AKN_JSON_PATH = OUTPUT_DIR / os.getenv("AKN_JSON_NAME", "struttura_akn.json")
AKN_XML_PATH = OUTPUT_DIR / os.getenv("AKN_XML_NAME", "documento_akn.xml")

VIRTUAL_ROOT_LABEL = "document"
# I livelli di DEFAULT: la mappa effettiva viene ricalcolata in build_tree()
# unendoli a quella dedotta dal documento (analysis["mappa_gerarchica"]), quindi
# l'insieme delle etichette gerarchiche vive li' come variabile locale.
DEFAULT_LEVELS = {k: v for k, v in AKN_LEVELS.items() if k != "recital"}
TERMINAL_NODE_LABELS = {"recital"}
STRUCTURE_BLOCKING_STATUSES = {
    "llm_json_error",
    "llm_review_missing",
    "pending_llm_review",
    "slm_failed_unresolved",
}
STRICT_ARTICLE_HEADERS = os.getenv("STRICT_ARTICLE_HEADERS", "1") == "1"
FUZZY_DEDUPE_THRESHOLD = float(os.getenv("FUZZY_DEDUPE_THRESHOLD", "0.97"))
ANNEX_LOCAL_PART_RE = LANGUAGE.patterns["annex_local_part"]

MIN_RAG_CHUNK_WORDS = int(os.getenv("MIN_RAG_CHUNK_WORDS", "60"))
MAX_RAG_CHUNK_WORDS = int(os.getenv("MAX_RAG_CHUNK_WORDS", "220"))
# Overlap (in parole) usato SOLO quando un singolo blocco troppo lungo va spezzato a
# forza in finestre da MAX_RAG_CHUNK_WORDS: senza sovrapposizione una frase-risposta a
# cavallo del taglio si perde in entrambi i pezzi. Non si applica al packing per
# paragrafo (che taglia a confini naturali e non ne ha bisogno). 0 = nessun overlap.
RAG_CHUNK_OVERLAP_WORDS = int(os.getenv("RAG_CHUNK_OVERLAP_WORDS", "40"))
RAG_CHUNK_LABEL = os.getenv("RAG_CHUNK_LABEL", "rag_chunk")

# Marcatori del preambolo e dell'articolato: dipendono dalla lingua del
# documento e arrivano dal profilo attivo (vedi legisleaf/language).
RECITAL_RE = LANGUAGE.patterns["recital_item"]
RECITAL_START_RE = LANGUAGE.patterns["recital_section_start"]
DEVICE_START_RE = LANGUAGE.patterns["device_start"]
PREAMBLE_MARKER_RE = LANGUAGE.patterns["preamble_marker"]
FOOTNOTE_CITATION_RE = LANGUAGE.patterns["footnote_citation_markers"]
RECITAL_SEQUENCE_TOLERANCE = int(os.getenv("RECITAL_SEQUENCE_TOLERANCE", "3"))
# Riusato per riconoscere se la rubrica di una presunta intestazione duplicata
# e' un titolo autonomo o la continuazione di una frase (vedi
# detect_repeated_headings, quarto caso).
ARTICLE_COMPONENTS_RE = LANGUAGE.patterns["article_components"]


def write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def testo(record: dict) -> str:
    return (record.get("testo") or record.get("text") or record.get("content") or "").strip()


def label(record: dict) -> str:
    return (record.get("etichetta") or "").lower().strip()


def confidence(record: dict):
    return record.get("label_confidence", record.get("confidence", record.get("confidenza")))


def record_sort_key(record: dict):
    page = record.get("page")
    page_order = record.get("page_order")
    ordine = record.get("ordine")
    split_index = record.get("split_index")
    return (
        page is None,
        page if page is not None else 0,
        page_order is None,
        page_order if page_order is not None else ordine if ordine is not None else 0,
        ordine if ordine is not None else 0,
        split_index if split_index is not None else 0,
    )


def dedupe_text_key(record: dict) -> str:
    text = record.get("testo")
    inferred = infer_structural_label(text)
    if inferred:
        number = extract_structural_number(text, inferred)
        if number:
            normalized_text = normalize_legal_text(text).lower()
            zone = record_zone(record)
            block_type = record_block_type(record)
            return f"{zone}|{block_type}|{inferred}:{number}|{normalized_text}".lower()
    return normalize_legal_text(text).lower()


def record_identity(record: dict) -> tuple:
    return (
        record.get("file"),
        record.get("page"),
        record.get("page_order"),
        record_zone(record),
        record_block_type(record),
        record.get("source", {}).get("source_model") if isinstance(record.get("source"), dict) else None,
        dedupe_text_key(record),
    )


def dedupe_records(records: list[dict]) -> list[dict]:
    priority = {
        "llm_reviewed": 6,
        "accepted_slm": 4,
        "deterministic": 3,
        "llm_json_error": 0,
        "llm_review_missing": 0,
        "pending_llm_review": 0,
        "slm_failed_unresolved": 0,
    }
    merged: dict[tuple, dict] = {}
    for record in sorted(records, key=record_sort_key):
        key = record_identity(record)
        old = merged.get(key)
        if old is None or priority.get(record.get("review_status"), 0) >= priority.get(old.get("review_status"), 0):
            merged[key] = record

    result: list[dict] = []
    for record in sorted(merged.values(), key=record_sort_key):
        normalized = dedupe_text_key(record)
        duplicate_index = None
        for index in range(len(result) - 1, -1, -1):
            candidate = result[index]
            if candidate.get("page") != record.get("page"):
                continue
            candidate_normalized = dedupe_text_key(candidate)
            if normalized and candidate_normalized and SequenceMatcher(None, normalized, candidate_normalized).ratio() >= FUZZY_DEDUPE_THRESHOLD:
                duplicate_index = index
                break
        if duplicate_index is None:
            result.append(record)
            continue
        old = result[duplicate_index]
        if priority.get(record.get("review_status"), 0) >= priority.get(old.get("review_status"), 0):
            result[duplicate_index] = record
    return result


def new_node(title: str, node_label: str, conf, source: dict) -> dict:
    number = extract_structural_number(title, node_label)
    node = {
        "name": f"{node_label}:{number}" if number else node_label,
        "intestazione": title,
        "etichetta": node_label,
        "numero": number,
        "confidenza": conf,
        "source": source,
        "contenuto": [],
        "figli": [],
    }
    if node_label == "article":
        _, rubrica = split_article_heading(title)
        if rubrica:
            node["rubrica"] = rubrica
    return node


def new_content(item_label: str, text: str, conf, source: dict) -> dict:
    return {
        "etichetta": item_label,
        "testo": text,
        "confidenza": conf,
        "source": source,
    }


def fix_article_label(text: str) -> str:
    normalized = normalize_legal_text(text)
    if PARAGRAPH_MARKER_RE.match(normalized):
        return "paragraph"
    if POINT_MARKER_RE.match(normalized):
        return "point"
    return "content"


def is_annex_local_part(text: str) -> bool:
    return bool(ANNEX_LOCAL_PART_RE.match(normalize_legal_text(text)))


def inside_annex(stack: list[dict]) -> bool:
    return any(node.get("etichetta") == "annex" for node in stack)


def split_structural_record(record: dict) -> list[dict]:
    text = testo(record)
    if not text or label(record) not in DEFAULT_LEVELS:
        return [record]

    matches = list(STRUCTURAL_MARKER_RE.finditer(text))
    if len(matches) <= 1 or matches[0].start() > 3:
        return [record]

    pieces = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        piece_text = text[start:end].strip()
        if not piece_text:
            continue
        piece = dict(record)
        piece["testo"] = piece_text
        piece["split_from_testo"] = text
        piece["split_index"] = index
        inferred = infer_structural_label(piece_text)
        if inferred:
            piece["etichetta"] = inferred
        pieces.append(piece)

    return pieces or [record]



# Un'intestazione ripetuta in cima a piu' pagine e' una testatina, non un nuovo
# nodo. Soglie misurate su data_act, dove "Schedule 2 - Exemptions etc from the
# GDPR" compariva 14 volte: una in indice e tredici come testatina delle pagine
# 162-177, tutte a page_order 1 o 2.
RUNNING_HEADER_MAX_PAGE_ORDER = int(os.getenv("RUNNING_HEADER_MAX_PAGE_ORDER", "2"))
RUNNING_HEADER_MIN_PAGES = int(os.getenv("RUNNING_HEADER_MIN_PAGES", "3"))
# Una pagina di indice e' fatta quasi solo di intestazioni. Su data_act la
# pagina 13 ne aveva 28 su 44 blocchi; nessun'altra pagina del corpus supera
# questa soglia.
CONTENTS_PAGE_MIN_HEADINGS = int(os.getenv("CONTENTS_PAGE_MIN_HEADINGS", "5"))
CONTENTS_PAGE_MIN_RATIO = float(os.getenv("CONTENTS_PAGE_MIN_RATIO", "0.5"))


def record_key(record: dict) -> tuple:
    return (record.get("file"), record.get("page"), record.get("page_order"), record.get("ordine"))


def _e_intestazione(record: dict) -> bool:
    """Il blocco ha la forma di un'intestazione strutturale?

    Calcolato una volta e passato in giro: ``infer_structural_label`` prova tutti
    i pattern del profilo, e i quattro criteri qui sotto lo interrogherebbero
    ognuno per ogni record.
    """
    text = testo(record)
    return bool(text) and bool(infer_structural_label(text))


def _pagine_di_indice_e_testatine(
    records: list[dict], intestazione: list[bool]
) -> tuple[set, set[str]]:
    """Le due grandezze da cui nascono i primi due criteri.

    ``pagine_di_indice``: pagine fatte quasi solo di intestazioni (l'indice).
    ``testatine``: testi che compaiono in CIMA a un numero minimo di pagine
    diverse, cioe' l'intestazione ripetuta dall'impaginazione.
    """
    blocchi_per_pagina: dict = defaultdict(int)
    intestazioni_per_pagina: dict = defaultdict(int)
    in_cima_alla_pagina: dict = defaultdict(set)

    for indice, record in enumerate(records):
        pagina = record.get("page")
        blocchi_per_pagina[pagina] += 1
        if not intestazione[indice]:
            continue
        intestazioni_per_pagina[pagina] += 1
        # Confronto esplicito con None: ``page_order`` vale 0 per il primo
        # blocco della pagina, e ``or 99`` lo scartava perche' zero e' falsy —
        # cioe' proprio le testatine in cima assoluta non venivano mai viste.
        page_order = record.get("page_order")
        if page_order is not None and page_order <= RUNNING_HEADER_MAX_PAGE_ORDER:
            in_cima_alla_pagina[normalize_legal_text(testo(record))].add(pagina)

    pagine_di_indice = {
        pagina
        for pagina, quante in intestazioni_per_pagina.items()
        if quante >= CONTENTS_PAGE_MIN_HEADINGS
        and quante / max(blocchi_per_pagina[pagina], 1) >= CONTENTS_PAGE_MIN_RATIO
    }
    testatine = {
        text for text, pagine in in_cima_alla_pagina.items()
        if len(pagine) >= RUNNING_HEADER_MIN_PAGES
    }
    return pagine_di_indice, testatine


def _sopprimi_indice_e_testatine(
    records: list[dict], intestazione: list[bool], pagine_di_indice: set, testatine: set[str]
) -> set[tuple]:
    """Criteri 1 e 2: la voce d'indice e le ripetizioni della testatina.

    Della testatina si tiene la PRIMA occorrenza fuori dall'indice: e' li' che la
    partizione comincia davvero. Tenere quella dell'indice collocherebbe il nodo
    decine di pagine prima del suo contenuto.
    """
    soppresse: set[tuple] = set()
    prima_tenuta: set[str] = set()
    for indice, record in enumerate(records):
        if not intestazione[indice]:
            continue
        normalizzato = normalize_legal_text(testo(record))
        if record.get("page") in pagine_di_indice:
            soppresse.add(record_key(record))
            continue
        if normalizzato in testatine:
            if normalizzato in prima_tenuta:
                soppresse.add(record_key(record))
            else:
                prima_tenuta.add(normalizzato)
    return soppresse


def _contenuto_dopo_ogni_intestazione(intestazione: list[bool]) -> list[int]:
    """Quanti blocchi NON-intestazione seguono ciascuna intestazione.

    E' il criterio che distingue la dichiarazione vera di una partizione dalle sue
    ripetizioni: l'intestazione vera e' seguita dal testo della partizione, la
    voce d'indice e il rimando sono seguiti subito da un'altra intestazione.
    """
    seguito: list[int] = []
    for indice in range(len(intestazione)):
        if not intestazione[indice]:
            seguito.append(0)
            continue
        quanti = 0
        cursore = indice + 1
        while cursore < len(intestazione) and not intestazione[cursore]:
            quanti += 1
            cursore += 1
        seguito.append(quanti)
    return seguito


def _intestazioni_per_chiave(
    records: list[dict], intestazione: list[bool]
) -> dict[tuple, list[int]]:
    """Indici dei record raggruppati per (etichetta, numero) della partizione.

    Due record nella stessa voce dichiarano la STESSA partizione: nel CFR
    "§ 164.512" compariva 8 volte per 41 numeri distinti contro 39 del golden; in
    data_act 990 intestazioni per 377 chiavi.
    """
    per_chiave: dict[tuple, list[int]] = defaultdict(list)
    for indice, record in enumerate(records):
        if not intestazione[indice]:
            continue
        text = testo(record)
        etichetta = infer_structural_label(text)
        numero = extract_structural_number(text, etichetta) or normalize_legal_text(text)
        per_chiave[(etichetta, numero)].append(indice)
    return per_chiave


def _sopprimi_duplicati_senza_corpo(
    records: list[dict], per_chiave: dict[tuple, list[int]], seguito: list[int],
    soppresse: set[tuple],
) -> int:
    """Criterio 3: fra piu' dichiarazioni della stessa partizione, tiene quelle col corpo.

    Si sopprime solo quando esiste almeno un'occorrenza con del testo dopo di se':
    se nessuna ce l'ha non c'e' un criterio per preferirne una, e una partizione
    realmente priva di testo non deve sparire. ``soppresse`` viene ESTESO in
    place, cosi' il criterio 4 vede gia' queste decisioni.
    """
    quanti = 0
    for indici in per_chiave.values():
        if len(indici) < 2:
            continue  # non e' un duplicato: una sola occorrenza
        if not any(seguito[indice] > 0 for indice in indici):
            continue
        for indice in indici:
            if seguito[indice] == 0:
                soppresse.add(record_key(records[indice]))
                quanti += 1
    return quanti


def _sembra_un_titolo(record: dict) -> bool:
    """La rubrica catturata comincia in maiuscolo (o non c'e' affatto)?

    E' cio' che distingue un'intestazione vera da un rimando finito a inizio
    blocco per l'OCR ("... under § 164.508 for use..."): un titolo autonomo
    comincia maiuscolo o non ha rubrica, una citazione a meta' frase continua in
    minuscolo. La forma della rubrica arriva da ``article_components``, che e'
    definito per lingua nel profilo.
    """
    match = ARTICLE_COMPONENTS_RE.match(normalize_legal_text(testo(record)))
    rubrica = ((match.group("rubrica") if match else "") or "").strip()
    return not rubrica or not rubrica[0].islower()


def _declassa_ri_intestazioni(
    records: list[dict], per_chiave: dict[tuple, list[int]], soppresse: set[tuple]
) -> set[tuple]:
    """Criterio 4: la ri-intestazione spuria si declassa, non si sopprime.

    Il testo di un rimando E' testo normativo, quindi non va perso: va solo
    impedito che apra un nodo spurio con lo stesso numero di uno gia' aperto.
    Diversamente dal criterio 3 il corpo seguente non aiuta — un rimando PUO'
    avere testo dopo di se' — e a decidere e' la forma della rubrica.
    """
    da_declassare: set[tuple] = set()
    for indici in per_chiave.values():
        if len(indici) < 2:
            continue
        canonico = next((i for i in indici if _sembra_un_titolo(records[i])), None)
        if canonico is None:
            # Nessuna occorrenza sembra un titolo autonomo: non c'e' un criterio
            # per scegliere quale tenere, restano tutte (stessa cautela del
            # criterio 3).
            continue
        for indice in indici:
            if indice == canonico or _sembra_un_titolo(records[indice]):
                continue
            chiave = record_key(records[indice])
            if chiave in soppresse:
                continue
            da_declassare.add(chiave)
    return da_declassare


def detect_repeated_headings(records: list[dict]) -> tuple[set[tuple], set[tuple], dict]:
    """Intestazioni che NON devono generare un nodo, per quattro criteri distinti.

    Un documento impaginato ripete la stessa intestazione in cima a ogni pagina
    della partizione, e ne elenca tutte in un indice iniziale. Per f2 sono
    intestazioni a tutti gli effetti — stesso testo, stessa forma — e ognuna
    creava un nodo: 336 nodi strutturali in data_act contro 151 testi distinti,
    con 14 nodi "Schedule 2" al posto di uno.

    Non si deduplica per solo testo: un allegato puo' legittimamente contenere una
    "Part 1" propria, gia' prevista dalla pipeline. A distinguere le ripetizioni
    servono quattro criteri, ognuno con il proprio segnale, applicati in
    quest'ordine (i primi tre sopprimono, il quarto declassa):

    1. **pagina d'indice** — densita' anomala di intestazioni sulla pagina;
    2. **testatina** — stesso testo in cima a piu' pagine diverse;
    3. **duplicato senza corpo** — stessa partizione dichiarata piu' volte, si
       tiene quella seguita dal testo;
    4. **ri-intestazione dentro una frase** — stessa partizione, ma la rubrica
       catturata continua in minuscolo: e' un rimando, resta come contenuto.

    Ritorna ``(soppresse, da_declassare, report)``.
    """
    intestazione = [_e_intestazione(record) for record in records]

    pagine_di_indice, testatine = _pagine_di_indice_e_testatine(records, intestazione)
    soppresse = _sopprimi_indice_e_testatine(records, intestazione, pagine_di_indice, testatine)

    seguito = _contenuto_dopo_ogni_intestazione(intestazione)
    per_chiave = _intestazioni_per_chiave(records, intestazione)
    duplicati_senza_corpo = _sopprimi_duplicati_senza_corpo(
        records, per_chiave, seguito, soppresse
    )
    da_declassare = _declassa_ri_intestazioni(records, per_chiave, soppresse)

    report = {
        "contents_pages": sorted(p for p in pagine_di_indice if p is not None),
        "running_header_texts": len(testatine),
        "empty_duplicate_headings": duplicati_senza_corpo,
        "suppressed_headings": len(soppresse),
        "duplicate_headings_downgraded_to_content": len(da_declassare),
    }
    return soppresse, da_declassare, report


ARTICLE_RUBRIC_POSITION = LANGUAGE.convention("article_rubric_position")
ARTICLE_RUBRIC_MAX_WORDS = int(LANGUAGE.convention("article_rubric_max_words"))
RUBRIC_BLOCK_TYPES = {str(t).lower() for t in LANGUAGE.convention("rubric_block_types")}
RUBRIC_LOOKAHEAD_BLOCKS = int(LANGUAGE.convention("rubric_lookahead_blocks"))
ARTICLE_BODY_OPENING_RE = LANGUAGE.patterns.get("article_body_opening")


def detect_rubric_pairs(records: list[dict]) -> tuple[dict[tuple, dict], set[tuple], dict]:
    """Ricompone le intestazioni spezzate su due blocchi.

    Vale per le lingue che dichiarano ``article_rubric_position: "before"``: nella
    redazione di common law la rubrica di una sezione e' una riga separata SOPRA
    il testo numerato, non sulla stessa riga del numero come in italiano.

        Superannuation of Commissioners      <- rubrica, nessun numero
        22. (1) The Commission shall...      <- numero, apre il corpo

    Vedendo due blocchi separati f2 non poteva riconoscere l'intestazione: su
    ``data_protection`` 221 rubriche su 232 sezioni finivano etichettate
    ``content``, e il numero restava sepolto in un blocco di testo. Le due parti
    insieme portano numero E rubrica, quindi il nodo che ne nasce non e' il nodo
    fantasma senza numero che ``apply_strict_structural_validation`` vuole
    prevenire: la fusione soddisfa quel vincolo, non lo aggira.

    Ritorna ``(articoli_forzati, rubriche_consumate, report)``.
    """
    if ARTICLE_RUBRIC_POSITION != "before" or ARTICLE_BODY_OPENING_RE is None:
        return {}, set(), {"attiva": False, "posizione_rubrica": ARTICLE_RUBRIC_POSITION}

    forzati: dict[tuple, dict] = {}
    consumate: set[tuple] = set()

    for index, record in enumerate(records):
        text = testo(record)
        if not text or record_block_type(record) not in RUBRIC_BLOCK_TYPES:
            continue
        if count_words(text) > ARTICLE_RUBRIC_MAX_WORDS or infer_structural_label(text):
            continue

        # primo blocco non vuoto entro la finestra dichiarata dal profilo
        seguente = None
        for offset in range(1, RUBRIC_LOOKAHEAD_BLOCKS + 1):
            if index + offset >= len(records):
                break
            candidato = records[index + offset]
            if testo(candidato):
                seguente = candidato
                break
        if seguente is None:
            continue

        match = ARTICLE_BODY_OPENING_RE.match(normalize_legal_text(testo(seguente)))
        if not match:
            continue
        numero = normalize_number(match.group("number"))
        if not numero:
            continue

        forzati[record_key(seguente)] = {"numero": numero, "rubrica": text.strip()}
        consumate.add(record_key(record))

    return forzati, consumate, {
        "attiva": True,
        "posizione_rubrica": ARTICLE_RUBRIC_POSITION,
        "coppie_ricomposte": len(forzati),
    }


def record_block_type(record: dict) -> str:
    source = record.get("source") or {}
    return (
        source.get("block_type_canonical")
        or source.get("block_type")
        or record.get("block_type_canonical")
        or record.get("block_type")
        or ""
    ).lower().strip()


def record_zone(record: dict) -> str:
    source = record.get("source") or {}
    return (source.get("document_zone") or record.get("document_zone") or "").lower().strip()


# Alcune fonti pubblicano ogni disposizione anche in un'altra lingua ufficiale
# (vedi language_en.json, sezione structure_conventions, per il caso EN/FR).
# Il marcatore che riconosce la lingua estranea e' un dato del profilo attivo,
# non del codice: un profilo che non lo dichiara (es. quello italiano) lascia
# la funzione sempre disattiva, cambiamento di comportamento zero.
_FOREIGN_TEXT_MARKER_PATTERN = LANGUAGE.convention("foreign_text_marker_pattern")
FOREIGN_TEXT_MARKER_RE = (
    re.compile(_FOREIGN_TEXT_MARKER_PATTERN, re.IGNORECASE) if _FOREIGN_TEXT_MARKER_PATTERN else None
)
FOREIGN_TEXT_MIN_MARKER_RATIO = float(LANGUAGE.convention("foreign_text_min_marker_ratio"))
FOREIGN_TEXT_MIN_WORDS = int(LANGUAGE.convention("foreign_text_min_words"))


def foreign_marker_ratio(text: str, marker_re: re.Pattern[str] | None) -> float:
    """Quota di PAROLE (non caratteri: un solo diacritico in un blocco lungo
    non basta) che contengono il marcatore. Funzione pura, presa un pattern
    esplicito cosi' e' testabile indipendentemente dal profilo attivo nel
    processo.

    Le parole che iniziano per maiuscola non contano: un nome proprio preso in
    prestito da un'altra lingua (istituzioni, toponimi) non rende straniera la
    frase inglese che lo contiene. Misurato su data_protection: 'Dail Eireann'
    e 'Seanad Eireann', citati piu' volte in paragrafi altrimenti tutti in
    inglese, bastavano a superare la soglia prima di questa esclusione."""
    if marker_re is None:
        return 0.0
    words = re.findall(r"\S+", text or "")
    if not words:
        return 0.0
    marked = sum(1 for w in words if not w[:1].isupper() and marker_re.search(w))
    return marked / len(words)


def is_foreign_language_block(text: str) -> bool:
    """Un blocco scritto in una lingua diversa da quella del documento attivo.

    Sotto ``foreign_text_min_words`` non si decide: un titolo troppo corto non
    da' un rapporto affidabile."""
    if FOREIGN_TEXT_MARKER_RE is None:
        return False
    words = re.findall(r"\S+", text or "")
    if len(words) < FOREIGN_TEXT_MIN_WORDS:
        return False
    return foreign_marker_ratio(text, FOREIGN_TEXT_MARKER_RE) >= FOREIGN_TEXT_MIN_MARKER_RATIO


def is_probable_footnote(record: dict, normalized_text: str) -> bool:
    block_type = record_block_type(record)
    if block_type in {"footnote", "page_footnote", "aside_text", "note", "docling_note", "mineru_note", "page_footer", "footer"}:
        return True
    if record_zone(record) in {"footnote", "footnotes", "notes", "headers_footers"}:
        return True

    if not RECITAL_RE.match(normalized_text):
        return False

    citation_markers = FOOTNOTE_CITATION_RE.search(normalized_text)
    return bool(citation_markers and count_words(normalized_text) <= 90)


def is_valid_recital_sequence(number: int, last_number: int | None) -> bool:
    if last_number is None:
        return number == 1
    return last_number <= number <= last_number + RECITAL_SEQUENCE_TOLERANCE

def count_words(text: str) -> int:
    return len(re.findall(r"\w+", text or "", flags=re.UNICODE))


def average_conf(items: list[dict]):
    """Confidenza media dei blocchi che ne dichiarano una; None se nessuno."""
    values = []
    for item in items:
        confidenza = item.get("confidenza")
        if isinstance(confidenza, (int, float)):
            values.append(confidenza)
    if not values:
        return None
    return sum(values) / len(values)


def split_long_content_item(item: dict) -> list[dict]:
    text = item.get("testo", "")
    words = re.findall(r"\S+", text)
    if len(words) <= MAX_RAG_CHUNK_WORDS:
        return [item]

    # Finestre da MAX parole che avanzano di (MAX - overlap): ogni finestra ripete
    # le ultime ``overlap`` parole della precedente, cosi' una frase a cavallo del
    # taglio resta intera in almeno un chunk.
    overlap = max(0, min(RAG_CHUNK_OVERLAP_WORDS, MAX_RAG_CHUNK_WORDS - 1))
    step = MAX_RAG_CHUNK_WORDS - overlap

    parts = []
    for start in range(0, len(words), step):
        part = dict(item)
        part["testo"] = " ".join(words[start : start + MAX_RAG_CHUNK_WORDS])
        part["split_from_long_block"] = True
        part["split_word_start"] = start
        part["split_overlap_words"] = overlap
        parts.append(part)
        if start + MAX_RAG_CHUNK_WORDS >= len(words):
            break
    return parts


def chunk_contents(items: list[dict], path: list[str]) -> list[dict]:
    chunks = []
    buffer = []
    words = 0

    # Un blocco troppo lungo viene prima spezzato in piu' pezzi: da qui in poi
    # si lavora sulla lista gia' espansa.
    expanded_items = []
    for item in items:
        for piece in split_long_content_item(item):
            expanded_items.append(piece)

    for item in expanded_items:
        item_words = count_words(item.get("testo", ""))
        if buffer and words + item_words > MAX_RAG_CHUNK_WORDS:
            chunks.append(buffer)
            buffer = []
            words = 0
        buffer.append(item)
        words += item_words
        if words >= MAX_RAG_CHUNK_WORDS:
            chunks.append(buffer)
            buffer = []
            words = 0

    if buffer:
        if chunks and words < MIN_RAG_CHUNK_WORDS:
            previous_words = sum(count_words(x.get("testo", "")) for x in chunks[-1])
            if previous_words + words <= MAX_RAG_CHUNK_WORDS:
                chunks[-1].extend(buffer)
            else:
                chunks.append(buffer)
        else:
            chunks.append(buffer)

    result = []
    for i, group in enumerate(chunks):
        text = "\n".join(x["testo"] for x in group if x.get("testo"))
        labels = defaultdict(int)
        sources = []
        for x in group:
            labels[x.get("etichetta", "content")] += 1
            if x.get("source"):
                sources.append(x["source"])
        result.append(
            {
                "etichetta": RAG_CHUNK_LABEL,
                "testo": text,
                "word_count": count_words(text),
                "labels": dict(labels),
                "confidenza": average_conf(group),
                "path": path,
                "chunk_index": i,
                "source_items": sources,
            }
        )
    return result


def rag_node(node: dict, parent_path: list[str] | None = None) -> dict:
    parent_path = parent_path or []
    title = node.get("intestazione") or node.get("etichetta") or "Nodo"
    path = parent_path + [title]
    contents = node.get("contenuto", [])
    children = node.get("figli", [])
    content_words = sum(count_words(x.get("testo", "")) for x in contents)

    out = {
        "intestazione": node.get("intestazione"),
        "rubrica": node.get("rubrica"),
        "etichetta": node.get("etichetta"),
        "confidenza": node.get("confidenza"),
        "source": node.get("source"),
        "path": path,
        "contenuto": [],
        "rag_chunks": [],
        "figli": [],
    }

    if contents and content_words < MIN_RAG_CHUNK_WORDS:
        key = "header_text" if children else "rag_text"
        out[key] = "\n".join(x["testo"] for x in contents if x.get("testo"))
        out["rag_word_count"] = content_words
        label_counts = defaultdict(int)
        for item in contents:
            label_counts[item.get("etichetta", "content")] += 1
        out["rag_labels"] = dict(label_counts)
    else:
        out["rag_chunks"] = chunk_contents(contents, path)

    for child in children:
        out["figli"].append(rag_node(child, path))

    return out


def visit(node: dict, depth: int = 0, report: dict | None = None) -> dict:
    if report is None:
        report = {"n_nodi": 0, "n_contenuti": 0, "max_depth": 0}
    report["n_nodi"] += 1
    report["n_contenuti"] += len(node.get("contenuto", []))
    report["max_depth"] = max(report["max_depth"], depth)
    for child in node.get("figli", []):
        visit(child, depth + 1, report)
    return report


def _override(record: dict, da: str, a: str, motivo: str, text: str,
              *, con_pagina: bool = False) -> dict:
    """Una voce di ``structural_overrides``: quale etichetta e' cambiata e perche'.

    Sette punti del ciclo di ``build_tree`` declassano un blocco, e tutti devono
    registrarlo nello stesso formato: il report di f2 e' l'unico posto in cui si
    vede *perche'* un nodo non e' stato creato, e una voce con campi diversi dalle
    altre non e' confrontabile con loro.

    ``con_pagina`` aggiunge la pagina subito dopo ``ordine``, come fanno i due siti
    che declassano una ripetizione tipografica — la' la pagina non e' un dettaglio
    ma il criterio stesso della decisione.
    """
    voce = {"file": record.get("file"), "ordine": record.get("ordine")}
    if con_pagina:
        voce["page"] = record.get("page")
    voce.update({"from": da, "to": a, "reason": motivo, "preview": text[:180]})
    return voce


def prepara_record() -> tuple[list[dict], dict[str, int]]:
    """I blocchi classificati, pronti per l'attraversamento, e i livelli gerarchici.

    Tre trasformazioni, in quest'ordine:

    1. **deduplica** — lo stesso blocco puo' arrivare due volte (da f1b e dalla
       revisione di f1c) e va tenuta la versione con lo stato di revisione
       migliore;
    2. **spezza** — un blocco che contiene piu' intestazioni concatenate diventa
       piu' record, cosi' da qui in poi vale "un record = un blocco";
    3. **ordina** — per pagina e posizione nella pagina: l'albero si costruisce
       nell'ordine di lettura, quindi l'ordine E' struttura.

    I livelli gerarchici partono dai default e vengono sovrascritti dalla mappa
    dedotta dal documento (``output_analysis.json``). Il ``recital`` non e' un
    livello di contenitore e resta fuori: e' un nodo terminale agganciato alla
    radice.
    """
    risultati = dedupe_records(read_json(CLASSIFICATION_RESULTS_PATH))

    record_espansi = []
    for record in risultati:
        for piece in split_structural_record(record):
            record_espansi.append(piece)
    risultati = record_espansi
    risultati.sort(key=record_sort_key)

    levels = dict(DEFAULT_LEVELS)
    analysis = read_json(ANALYSIS_JSON_PATH)
    for nome_livello, profondita in analysis["mappa_gerarchica"].items():
        etichetta = nome_livello.lower()
        if etichetta == "recital":
            continue
        levels[etichetta] = int(profondita)

    return risultati, levels


def build_tree() -> tuple[list[dict], list[dict], dict]:
    """Dai blocchi classificati ai cinque artefatti di f2.

    Quattro fasi: si preparano i record (``prepara_record``), si individuano le
    intestazioni che non devono generare un nodo (``detect_repeated_headings``,
    ``detect_rubric_pairs``), si attraversano i blocchi nell'ordine di lettura
    costruendo l'albero, e infine si converte in Akoma Ntoso e si scrive
    (``scrivi_artefatti``).

    Il ciclo centrale resta un blocco unico di proposito: e' una macchina a stati
    sequenziale (dentro/fuori i considerando, articolato iniziato, primo articolo
    visto, ultimo numero di considerando, pila dei contenitori aperti) in cui ogni
    decisione dipende da quelle prese sui blocchi precedenti. Spezzarlo
    richiederebbe di far viaggiare quello stato fra le parti, cioe' di sostituire
    una sequenza leggibile con un passaggio di stato implicito.
    """
    risultati, levels = prepara_record()
    hierarchical_labels = set(levels)
    allowed_labels = hierarchical_labels | TERMINAL_NODE_LABELS | CONTENT_LABELS | NOISE_LABELS

    suppressed_headings, downgrade_to_content, repeated_report = detect_repeated_headings(risultati)
    forced_articles, consumed_rubrics, rubric_report = detect_rubric_pairs(risultati)

    print("Blocchi classificati:", len(risultati))
    print("Intestazioni soppresse (indice/testatine):", repeated_report)
    print("Intestazioni ricomposte (rubrica + numero):", rubric_report)
    print("Label gerarchiche:", levels)
    print("Label ammesse:", sorted(allowed_labels))

    root = new_node(
        DOCUMENT_NAME,
        VIRTUAL_ROOT_LABEL,
        None,
        {
            "classification_file": str(CLASSIFICATION_RESULTS_PATH),
            "analysis_file": str(ANALYSIS_JSON_PATH),
            "document_id": DOCUMENT_ID,
            "document_name": DOCUMENT_NAME,
        },
    )
    stack: list[dict] = []
    last_terminal_node: dict | None = None
    in_recitals = False
    device_started = False
    # I considerando stanno nel preambolo, PRIMA dell'articolato: nessun atto ne
    # ha dopo il primo articolo. Senza questo vincolo le note a pie' di pagina
    # numerate "(N)" venivano prese per considerando ogni volta che il marcatore
    # di fine preambolo non veniva riconosciuto: 67 su 69 in codice_consumo,
    # tutti e 21 in codice_contratti, ma anche 34 in nis2 e 10 in dora, dove
    # erano rimandi bibliografici ("(27) Direttiva 2011/93/UE...").
    first_article_seen = False
    demoted_recitals: list[dict] = []
    last_recital_number: int | None = None
    unknown_labels = defaultdict(int)
    article_fixes = []
    structural_overrides = []
    review_missing = []
    foreign_language_blocks = []

    for record in risultati:
        text = testo(record)
        if not text:
            continue

        current_label = label(record)
        original_label = current_label
        conf = confidence(record)
        normalized_text = normalize_legal_text(text)
        review_status = record.get("review_status")

        if current_label == "unresolved" or review_status in STRUCTURE_BLOCKING_STATUSES:
            review_missing.append(record)
            continue

        # Blocco nell'altra lingua di una fonte bilingue (vedi
        # is_foreign_language_block): esce PRIMA di ogni altra decisione,
        # cosi' non genera ne' un nodo strutturale duplicato ne' un chunk RAG
        # ridondante. Disattivo per i profili che non dichiarano il marcatore.
        if is_foreign_language_block(text):
            foreign_language_blocks.append(
                {
                    "file": record.get("file"),
                    "ordine": record.get("ordine"),
                    "page": record.get("page"),
                    "preview": text[:180],
                }
            )
            continue

        if RECITAL_START_RE.match(normalized_text):
            in_recitals = True
            device_started = False
            last_recital_number = None
            current_label = "content"

        if DEVICE_START_RE.match(normalized_text):
            in_recitals = False
            device_started = True
            last_terminal_node = None

        recital_match = RECITAL_RE.match(normalized_text)
        recital_validated = False
        probable_footnote = is_probable_footnote(record, normalized_text)
        if (
            recital_match
            and in_recitals
            and not device_started
            and not first_article_seen
            and not probable_footnote
        ):
            recital_number = int(recital_match.group(1))
            if is_valid_recital_sequence(recital_number, last_recital_number):
                current_label = "recital"
                last_recital_number = recital_number
                recital_validated = True

        # Una predizione recital è accettata solo se confermata dallo stato documentale.
        if current_label == "recital" and not recital_validated:
            if first_article_seen:
                demoted_recitals.append(
                    {
                        "file": record.get("file"),
                        "ordine": record.get("ordine"),
                        "page": record.get("page"),
                        "reason": "recital_after_first_article",
                        "preview": text[:180],
                    }
                )
            current_label = "noise" if probable_footnote else "content"

        if PREAMBLE_MARKER_RE.match(normalized_text) and not RECITAL_RE.match(normalized_text):
            current_label = "content"

        chiave = record_key(record)

        # La rubrica e' stata fusa nel blocco numerato che la segue: qui
        # genererebbe un secondo nodo, o resterebbe testo orfano.
        if chiave in consumed_rubrics:
            continue

        # Il blocco apre una sezione ed eredita numero e rubrica dal blocco
        # precedente: e' un'intestazione a tutti gli effetti.
        if chiave in forced_articles:
            current_label = "article"

        # Indice e testatine: il testo c'e' gia' altrove nel documento, e qui
        # genererebbe un nodo duplicato che nessun golden puo' abbinare.
        if record_key(record) in suppressed_headings:
            structural_overrides.append(
                _override(record, original_label, "noise",
                          "repeated_heading_index_or_running_header", text, con_pagina=True)
            )
            continue

        inferred_label = infer_structural_label(text)
        # Il pattern "(N) " che riconosce un considerando e' lo stesso di una nota
        # a pie' di pagina: a distinguerli e' solo la posizione nel documento.
        # Dopo il primo articolo il preambolo e' chiuso, quindi l'override
        # deterministico non puo' piu' promuovere a considerando - altrimenti
        # rimetterebbe l'etichetta proprio sui blocchi che la macchina a stati
        # ha appena rifiutato, annullandone la decisione. Prima del primo
        # articolo l'override resta attivo: e' quello che riconosce i
        # considerando veri degli atti UE.
        if inferred_label == "recital" and first_article_seen:
            inferred_label = None
        if (
            inferred_label
            and current_label != inferred_label
            and current_label != "recital"
            and not PREAMBLE_MARKER_RE.match(normalized_text)
            and not text.lstrip().startswith("<")
        ):
            current_label = inferred_label
            structural_overrides.append(
                _override(record, original_label, current_label,
                          "deterministic_structural_header", text)
            )

        if current_label == "part" and inside_annex(stack) and is_annex_local_part(text):
            current_label = "content"
            structural_overrides.append(
                _override(record, original_label, current_label,
                          "local_part_inside_annex", text)
            )

        if current_label in levels and is_structural_editorial_note(text):
            current_label = "content"
            structural_overrides.append(
                _override(record, original_label, current_label,
                          "structural_editorial_note", text)
            )

        if current_label not in allowed_labels:
            unknown_labels[current_label or "<vuota>"] += 1
            current_label = "content"

        if record.get("review_status") == "llm_review_missing":
            review_missing.append(record)

        # L'articolo ricomposto porta numero E rubrica dalla fusione con il blocco
        # precedente: il controllo strict serve a impedire i nodi senza numero, e
        # qui il numero c'e'. Applicarlo anche a questi declasserebbe proprio le
        # intestazioni che la fusione ha appena ricostruito.
        if STRICT_ARTICLE_HEADERS and current_label == "article" and chiave not in forced_articles:
            if not is_article_header(normalized_text):
                current_label = fix_article_label(text)
                article_fixes.append(
                    {
                        "file": record.get("file"),
                        "ordine": record.get("ordine"),
                        "from": original_label,
                        "to": current_label,
                        "normalized_preview": normalized_text[:180],
                        "preview": text[:180],
                    }
                )

        if (
            current_label in hierarchical_labels
            and current_label != "article"
            and infer_structural_label(text) != current_label
        ):
            structural_overrides.append(
                _override(record, original_label, "content",
                          "strict_structural_header_failed", text)
            )
            current_label = "content"

        if (
            (record.get("source") or {}).get("recovered_extra_candidate")
            and current_label in (hierarchical_labels | TERMINAL_NODE_LABELS)
            and review_status != "llm_reviewed"
        ):
            structural_overrides.append(
                _override(record, original_label, "content",
                          "extra_candidate_requires_additional_validation", text)
            )
            current_label = "content"

        # Ultima parola: una ri-intestazione spuria (stesso numero, rubrica che
        # continua in minuscolo) resta 'content' anche se qualche passo sopra
        # l'avesse rimessa a un'etichetta gerarchica.
        if record_key(record) in downgrade_to_content and current_label != "content":
            structural_overrides.append(
                _override(record, original_label, "content",
                          "repeated_heading_body_continuation", text, con_pagina=True)
            )
            current_label = "content"

        source = {
            "file": record.get("file"),
            "ordine": record.get("ordine"),
            "page": record.get("page"),
            "page_order": record.get("page_order"),
            "review_status": record.get("review_status"),
            "etichetta_originale": original_label,
            "testo_normalizzato": normalized_text,
            "split_index": record.get("split_index"),
            "split_from_testo": record.get("split_from_testo"),
        }
        if isinstance(record.get("source"), dict):
            source.update(record["source"])

        if current_label in NOISE_LABELS:
            continue

        if current_label in hierarchical_labels:
            last_terminal_node = None
            if current_label == "article":
                first_article_seen = True
            node = new_node(text, current_label, conf, source)
            ricomposto = forced_articles.get(chiave)
            if ricomposto:
                node["numero"] = ricomposto["numero"]
                node["name"] = f"article:{ricomposto['numero']}"
                node["rubrica"] = ricomposto["rubrica"]
                node["intestazione"] = f"{ricomposto['numero']} {ricomposto['rubrica']}"
                node["source"]["rubrica_ricomposta"] = True
            node_level = levels[current_label]
            while stack and levels.get(stack[-1]["etichetta"], 99) >= node_level:
                stack.pop()
            parent = stack[-1] if stack else root
            parent["figli"].append(node)
            stack.append(node)

        elif current_label in TERMINAL_NODE_LABELS:
            node = new_node(text, current_label, conf, source)
            root["figli"].append(node)
            last_terminal_node = node

        else:
            if (
                current_label == "content"
                and stack
                and stack[-1].get("etichetta") == "article"
                and not stack[-1].get("rubrica")
                and not stack[-1].get("contenuto")
                and not stack[-1].get("figli")
                and count_words(text) <= 30
                and not infer_structural_label(text)
            ):
                stack[-1]["rubrica"] = text
                stack[-1]["rubrica_source"] = source
                continue

            item = new_content(current_label, text, conf, source)
            if in_recitals and last_terminal_node is not None:
                last_terminal_node["contenuto"].append(item)
            elif stack:
                stack[-1]["contenuto"].append(item)
            else:
                root["contenuto"].append(item)

    radici = [root]
    radici_rag = [rag_node(x) for x in radici]

    diagnostica = {
        "unknown_labels": unknown_labels,
        "article_fixes": article_fixes,
        "structural_overrides": structural_overrides,
        "review_missing": review_missing,
        "foreign_language_blocks": foreign_language_blocks,
        "demoted_recitals": demoted_recitals,
        "repeated_report": repeated_report,
        "rubric_report": rubric_report,
    }
    return scrivi_artefatti(root, radici, radici_rag, diagnostica)


def scrivi_artefatti(
    root: dict,
    radici: list[dict],
    radici_rag: list[dict],
    diagnostica: dict[str, Any],
) -> tuple[list[dict], list[dict], dict]:
    """Converte in Akoma Ntoso, assembla il report e scrive i cinque artefatti.

    ``diagnostica`` raccoglie ciò che il ciclo ha osservato strada facendo
    (declassamenti, override, blocchi scartati): sono liste di sole letture a
    questo punto, quindi la fase è pura riorganizzazione di dati.

    La conversione in Akoma Ntoso parte dalla STESSA radice dell'albero interno:
    i due artefatti non possono descrivere documenti diversi perché il secondo
    è generato dal primo, non ricostruito in parallelo.
    """
    akn_identity = akn_identity_from_document(root, DOCUMENT_ID, DOCUMENT_NAME)
    akn_tree = build_akn_tree(root, akn_identity)
    akn_xml = akn_tree_to_xml(akn_tree)
    akn_report = check_akn_tree(akn_tree)
    akn_report["identification"] = akn_tree["identification"]
    akn_report["xsd"] = validate_against_xsd(akn_xml)

    unknown_labels = diagnostica["unknown_labels"]
    article_fixes = diagnostica["article_fixes"]
    structural_overrides = diagnostica["structural_overrides"]
    foreign_language_blocks = diagnostica["foreign_language_blocks"]

    # Di ogni categoria si tengono i conteggi COMPLETI e i primi 25 esempi: il
    # report serve a capire cosa e' stato declassato e perche', non a contenere
    # ogni singolo caso (su data_act sarebbero migliaia di righe).
    validation_report = visit(root)
    validation_report["n_radici"] = len(radici)
    validation_report["unknown_labels"] = dict(unknown_labels)
    validation_report["article_fixes_count"] = len(article_fixes)
    validation_report["article_fixes_examples"] = article_fixes[:25]
    validation_report["structural_overrides_count"] = len(structural_overrides)
    validation_report["structural_overrides_examples"] = structural_overrides[:25]
    validation_report["review_missing_count"] = len(diagnostica["review_missing"])
    validation_report["foreign_language_blocks_count"] = len(foreign_language_blocks)
    validation_report["foreign_language_blocks_examples"] = foreign_language_blocks[:25]
    validation_report["recitals_demoted_after_first_article"] = len(diagnostica["demoted_recitals"])
    validation_report["recitals_demoted_examples"] = diagnostica["demoted_recitals"][:25]
    validation_report["terminal_node_labels"] = sorted(TERMINAL_NODE_LABELS)
    validation_report["recital_state_machine"] = True
    validation_report["repeated_headings"] = diagnostica["repeated_report"]
    validation_report["rubric_pairs"] = diagnostica["rubric_report"]
    validation_report["akn"] = akn_report
    validation_report["warnings"] = []

    if not root["figli"] and not root["contenuto"]:
        validation_report["warnings"].append("Albero vuoto.")
    if not akn_report.get("valido"):
        validation_report["warnings"].append("Albero AKN non conforme: vedi report['akn'].")

    write_json(FINAL_JSON_PATH, radici)
    write_json(RAG_JSON_PATH, radici_rag)
    write_json(AKN_JSON_PATH, akn_tree)
    write_bytes(AKN_XML_PATH, akn_xml)
    write_json(TREE_VALIDATION_REPORT_PATH, validation_report)

    print("Figli diretti documento:", len(root["figli"]))
    print("Contenuti diretti documento:", len(root["contenuto"]))
    print("Label ignote convertite in content:", dict(unknown_labels))
    print("Article corretti:", len(article_fixes))
    print("Override strutturali:", len(structural_overrides))
    print("Blocchi in lingua diversa dal documento:", len(foreign_language_blocks))
    print("Albero salvato:", FINAL_JSON_PATH)
    print("Albero RAG salvato:", RAG_JSON_PATH)
    print("Albero AKN salvato:", AKN_JSON_PATH)
    print("XML AKN salvato:", AKN_XML_PATH)
    print("Elementi AKN:", akn_report["conteggio_elementi"])
    print(
        f"Identita' AKN: {akn_identity.work_uri} "
        f"(tipo: {akn_identity.subtype or akn_identity.doctype}, "
        f"riconoscimento: {akn_identity.metadata_source})"
    )
    print("Report:", validation_report)
    return radici, radici_rag, validation_report


def main() -> None:
    build_tree()


if __name__ == "__main__":
    main()
