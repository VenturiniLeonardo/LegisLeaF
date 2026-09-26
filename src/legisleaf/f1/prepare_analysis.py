from __future__ import annotations

import json
import os
import hashlib
import re
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI

from sentence_transformers import SentenceTransformer
from legisleaf.model_manager import stop_big_model
from legisleaf.common import (
    AKN_LEVELS,
    AKN_PATTERNS,
    ALLOWED_LABELS,
    LANGUAGE,
    PAGE_NUMBER_NOISE_RE,
    PARAGRAPH_MARKER_RE,
    POINT_MARKER_RE,
    F1Config,
    allowed_text,
    infer_structural_label,
    load_blocks,
    load_config,
    load_json,
    valid_labels,
    write_json,
)


def build_sample_text(blocks: list[dict], config: F1Config) -> tuple[list[int], str]:
    pages = sorted({block["page"] for block in blocks if block["page"] is not None})
    if not pages:
        selected_pages = []
    else:
        n = min(max(config.n_sample_pages, 1), len(pages))
        selected = set()
        by_label: dict[str, int] = {}
        for block in blocks:
            page = block.get("page")
            if page is None:
                continue
            inferred = infer_structural_label(block.get("testo"))
            if inferred and inferred not in by_label:
                by_label[inferred] = page
        for label_name in ("article", "title", "chapter", "section", "annex", "recital"):
            if label_name in by_label:
                selected.add(by_label[label_name])
        positions = np.linspace(0, len(pages) - 1, n)
        selected.update(pages[int(round(pos))] for pos in positions)
        selected_pages = sorted(selected)

    sample_rows = []
    for block in blocks:
        if block["page"] in selected_pages:
            sample_rows.append(
                "[page={page} order={order} tipo_ocr={tipo}] {testo}".format(
                    page=block["page"],
                    order=block["page_order"],
                    tipo=block["block_type"],
                    testo=block["testo"],
                )
            )

    kept_rows = []
    current_chars = 0
    for row in sample_rows:
        row_len = len(row) + 1
        if kept_rows and current_chars + row_len > config.max_llm_sample_chars:
            break
        kept_rows.append(row)
        current_chars += row_len

    sample_text = "\n".join(kept_rows)
    if len(kept_rows) < len(sample_rows):
        sample_text += "\n[... campione troncato a confine blocco ...]"

    print("Pagine campionate:", selected_pages)
    print("Caratteri campione:", len(sample_text))
    print(sample_text[:1200])
    return selected_pages, sample_text


def build_analysis_prompts(sample_text: str, config: F1Config) -> tuple[str, str]:
    akn_patterns_text = json.dumps(AKN_PATTERNS, ensure_ascii=False, indent=2)
    akn_levels_text = json.dumps(AKN_LEVELS, ensure_ascii=False, indent=2)
    labels_text = allowed_text()
    # Esempi e glosse arrivano dal profilo linguistico: il prompt resta in
    # italiano (lingua operativa della pipeline), ma tutto cio' che descrive la
    # forma del documento segue la lingua del documento riconosciuta da f0.
    lingua = LANGUAGE.name
    glosse = LANGUAGE.hierarchy_glosses_text()
    es_article = LANGUAGE.examples_text("article")
    es_article_corti = LANGUAGE.examples_text("article", limit=4)
    es_annex = LANGUAGE.examples_text("annex")
    es_title = LANGUAGE.examples_text("title", limit=1)
    es_chapter = LANGUAGE.examples_text("chapter")
    es_section = LANGUAGE.examples_text("section")
    es_paragraph = LANGUAGE.examples_text("paragraph")
    es_point = LANGUAGE.examples_text("point")
    es_modificative = LANGUAGE.modificative_examples_text()
    es_citazioni = LANGUAGE.article_citation_examples_text()
    sinonimi_vietati = LANGUAGE.forbidden_synonyms_text()

    system_prompt = f"""Sei un Architetto dei Dati specializzato in ingegneria della conoscenza, diritto amministrativo e modellazione Akoma Ntoso.
Il tuo compito e' analizzare un campione estratto da un testo normativo in lingua {lingua} e produrre regole tipografiche locali e un golden set compatto per classificare blocchi OCR.

REGOLA FONDAMENTALE SULLE LABEL:
- I metadati visuali dell'estrattore (Tipo visivo, block_type, section_header, list_item, text, article_header, title, plain_text, picture) sono solo indizi grafici.
- Non devono mai comparire come etichette in label_rilevanti o golden_set.
- Le etichette finali devono descrivere la funzione normativa/strutturale del blocco secondo Akoma Ntoso.
- Puoi usare SOLO le etichette ammesse fornite dall'utente. Non inventare sinonimi, traduzioni o nuove label.
- Se sei incerto tra piu' label, scegli la piu' prudente: content per testo normativo ordinario, noise per materiale non normativo.

---

ANALISI RICHIESTA

1. Mappa Gerarchica Akoma Ntoso
   La mappa gerarchica e' gia' fissata dal codice e non va modificata:
{glosse}
   Non includere paragraph, point, content o noise nella mappa gerarchica.
   Non usare label visuali come section_header, list_item, text, article_header, title OCR o plain_text.

2. Struttura Intra-articolare
   Analizza gli elementi interni agli articoli e descrivi:
   - formato dei paragraph, cioe' i commi numerati come {es_paragraph};
   - formato dei point, cioe' lettere o punti di elenco come {es_point};
   - quando usare content come fallback normativo ordinario.

3. Pattern di Classificazione
   I pattern sono gia' fissati dal codice per evitare etichette inventate.
   Devi usarli come riferimento concettuale mentre scegli gli esempi.
   REGOLE:
   - article deve indicare solo intestazioni autonome del tipo {es_article};
   - title, chapter e section accettano numeri romani o arabi: {es_title}, {es_chapter}, {es_section};
   - annex deve indicare intestazioni autonome di allegato, per esempio {es_annex};
   - non classificare come article un paragraph che cita un articolo, per esempio {es_citazioni};
   - non classificare come struttura le istruzioni modificative: {es_modificative} sono content;
   - paragraph e' un comma numerato dentro un articolo;
   - point e' una voce di elenco;
   - content e' il fallback quando nessun livello strutturale o intra-articolare specifico e' riconosciuto;
   - noise e' materiale editoriale, intestazione, numero pagina, pubblicita', errore OCR o testo non normativo.

4. Label rilevanti
   Identifica le label ammesse piu' rilevanti per il retrieval esempi.
   Considera rilevanti le label frequenti, ambigue, centrali per la gerarchia o difficili da distinguere.
   Non usare label fuori dalla lista ammessa e non usare label visuali.
   Salvale nel campo label_rilevanti come lista di stringhe esattamente uguali alle label ammesse.

5. Golden Set
   Genera un golden set compatto per il database embeddings.
   REGOLE:
   - fornisci almeno 1 esempio per ogni label ammessa se presente nel campione;
   - fornisci 2 esempi per ogni label_rilevante, se presente nel campione;
   - non superare {config.n_llm_examples} esempi totali;
   - ogni testo_originale deve essere copiato verbatim dal campione;
   - ogni testo_originale deve contenere almeno 20 caratteri quando possibile;
   - golden_set[*].etichetta deve appartenere solo alla lista di etichette ammesse;
   - se una label non ha esempi nel campione, non inventare testo.

---

VINCOLO DI OUTPUT

Restituisci ESCLUSIVAMENTE un oggetto JSON valido.
- Nessun testo prima o dopo il JSON.
- Nessuna formattazione markdown.
- Nessun commento nel JSON.
- Usa solo label presenti nella lista ammessa.
- Rispetta esattamente lo schema richiesto dall'utente.
"""

    user_prompt = f"""Ecco un campione compatto di blocchi estratti dal documento normativo.
Ogni riga puo' contenere metadati come page, order, ID, Tipo visivo e tipo_ocr: questi metadati descrivono l'estrazione grafica, NON sono etichette strutturali finali.

Analizzalo e produci mappa gerarchica, regole tipografiche, pattern regex e golden set.
Nota importante: per evitare che LLM e SLM inventino etichette, in questa versione la mappa gerarchica e i pattern regex Akoma Ntoso sono fissati dal codice e vengono mostrati qui sotto. Usali come riferimento obbligatorio; non modificarli e non aggiungere label.

Etichette Akoma Ntoso ammesse, senza eccezioni:
{labels_text}

Mappa gerarchica fissata dal codice e salvata nell'output finale:
{akn_levels_text}

Pattern regex fissati dal codice e salvati nell'output finale:
{akn_patterns_text}

Vincoli sulle label:
- NON usare mai come etichette finali label visuali dell'estrattore come section_header, list_item, text, article_header, title, plain_text, picture;
- usa solo label strutturali normative Akoma Ntoso presenti nella lista ammessa;
- le label gerarchiche ammesse sono: part, annex, title, chapter, section, article;
- le label intra-articolari e di contenuto ammesse sono: paragraph, point, content, noise;
- non usare sinonimi in lingua {lingua} come {sinonimi_vietati};
- se il testo e' normativo ma non riconosciuto come struttura specifica, usa content;
- se il testo e' editoriale, rumore OCR, numero pagina o non normativo, usa noise.

Golden set richiesto:
- identifica tu quali label finali sono piu' rilevanti, frequenti o ambigue nel campione;
- salva queste label in label_rilevanti;
- fornisci almeno 1 esempio per ogni label gerarchica Akoma Ntoso presente nel campione;
- includi esempi article in formati diversi se presenti: {es_article_corti};
- fornisci almeno 1 esempio per ogni label intra-articolare o di contenuto presente nel campione: paragraph, point, content, noise;
- fornisci 2 esempi per ogni label che hai inserito in label_rilevanti, se presente nel campione;
- non superare {config.n_llm_examples} esempi totali;
- copia ogni testo_originale verbatim dal campione;
- ogni testo_originale deve contenere almeno 20 caratteri quando possibile;
- se una label non ha esempi nel campione, non inventare testo.

Restituisci SOLO questo JSON, senza testo prima o dopo:
{{
  "label_rilevanti": ["<una label ammessa>"],
  "regole_tipografiche": {{
    "formato_commi": "<descrizione sintetica>",
    "formato_liste": "<descrizione sintetica>"
  }},
  "golden_set": [
    {{
      "testo_originale": "<testo esatto estratto dal campione>",
      "etichetta": "<una label ammessa>",
      "ragionamento": "<motivazione sintetica>",
      "confidenza_attesa": <float tra 0.0 e 1.0>
    }}
  ]
}}

--- INIZIO CAMPIONE ---
{sample_text}
--- FINE CAMPIONE ---
"""

    print("Prompt pronto. Label bloccate:", sorted(ALLOWED_LABELS))
    return system_prompt, user_prompt


def analysis_cache_signature(config: F1Config, sample_text: str, system_prompt: str, user_prompt: str) -> dict:
    blocks_hash = hashlib.sha256(config.blocks_json_path.read_bytes()).hexdigest() if config.blocks_json_path.exists() else None
    return {
        "ocr_sha256": blocks_hash,
        "sample_sha256": hashlib.sha256(sample_text.encode("utf-8")).hexdigest(),
        "prompt_sha256": hashlib.sha256((system_prompt + "\n" + user_prompt).encode("utf-8")).hexdigest(),
        "regex_sha256": hashlib.sha256(json.dumps(AKN_PATTERNS, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
        "model_name": config.model_name,
        # Un profilo linguistico diverso cambia regex, esempi e prompt: l'analisi
        # in cache non e' piu' valida anche se OCR e modello sono gli stessi.
        "language": LANGUAGE.fingerprint(),
        "schema_version": 2,
    }


def load_cached_analysis(config: F1Config, signature: dict) -> dict | None:
    if not config.use_cached_analysis or not config.analysis_json_path.exists():
        return None
    with config.analysis_json_path.open("r", encoding="utf-8") as handle:
        cached = json.load(handle)
    ok = (
        cached.get("mappa_gerarchica") == AKN_LEVELS
        and valid_labels(cached.get("golden_set", []), "etichetta")
        and set(cached.get("label_rilevanti", [])) <= ALLOWED_LABELS
        and cached.get("cache_signature") == signature
    )
    if ok:
        print("[cache] Analisi caricata da", config.analysis_json_path)
        return cached
    print("[cache] Ignorata: label non AKN oppure mappa diversa.")
    return None


# Un backslash in JSON apre una sequenza di escape: sono ammessi solo
# " \ / b f n r t e \uXXXX con quattro cifre esadecimali. Il resto e' un errore
# di sintassi. La \u va verificata per intero, altrimenti "\usepackage" sembra
# un escape lecito e la riparazione lo lascia rotto.
_ESCAPE_VALIDI = set('"\\/bfnrt')
_SEQUENZA_BACKSLASH = re.compile(r"\\+")


def ripara_escape_json(raw: str) -> str:
    """Chiude le sequenze di escape non valide lasciando pari i backslash.

    Si ragiona per SEQUENZE di backslash, non un carattere alla volta. La
    versione precedente scandiva singolarmente e su ``\\\\ `` (a capo LaTeX: due
    backslash veri seguiti da spazio) si comportava cosi': il primo backslash e'
    seguito da un altro backslash, che e' un escape valido, quindi lo saltava; il
    secondo e' seguito da spazio, non valido, quindi lo raddoppiava. Ne uscivano
    TRE backslash, cioe' ``\\\\`` valido piu' ``\\ `` di nuovo illegale: la
    riparazione produceva il guasto che doveva togliere. Visto davvero su
    arxiv_1401_8087, dentro ``$\\begin{array}...\\\\ \\end{array}$``.

    La regola giusta guarda la parita': una sequenza di N backslash consuma
    N//2 coppie; se N e' pari il carattere che segue e' testo normale e non c'e'
    niente da fare, se N e' dispari l'ultimo backslash apre un escape sul
    carattere seguente, e va raddoppiato solo quando quell'escape non e' ammesso.

    Resta parziale per costruzione, come prima: ``\\ref`` e ``\\times`` sono
    indistinguibili da un ritorno a capo e da una tabulazione veri, perche'
    ``\\r`` e ``\\t`` SONO escape validi. Quei casi si perdono comunque; qui si
    recuperano tutti gli altri e soprattutto si evita che un documento faccia
    fallire l'intero step.
    """

    def chiudi(match: re.Match) -> str:
        sequenza = match.group(0)
        if len(sequenza) % 2 == 0:
            return sequenza
        dopo = raw[match.end():match.end() + 5]
        if dopo[:1] in _ESCAPE_VALIDI:
            return sequenza
        if dopo[:1] == "u" and re.fullmatch(r"[0-9a-fA-F]{4}", dopo[1:5] or ""):
            return sequenza
        return sequenza + "\\"

    return _SEQUENZA_BACKSLASH.sub(chiudi, raw)


def parse_llm_json(raw: str, rigenera=None) -> dict:
    """Legge il JSON del modello tollerando i backslash non escapati.

    Lo schema di questa richiesta ha campi di testo libero (``testo_originale``,
    ``ragionamento``) che ricopiano il documento alla lettera. Su un paper con
    formule il documento contiene LaTeX, e ``\\mathcal`` finito dentro una
    stringa JSON e' una sequenza di escape non valida: il decoding guidato
    vincola la STRUTTURA del JSON, non la corretta escapatura del testo che il
    modello ci mette dentro. Senza tolleranza un solo backslash fa fallire
    l'intero step (visto su arxiv_1401_8087).

    Ordine dei rimedi: prima si richiede la risposta al modello (``rigenera``),
    che con un campionamento diverso spesso escapa correttamente e restituisce il
    testo INTATTO; solo se anche il secondo tentativo e' rotto si ripara, perche'
    la riparazione puo' alterare il testo degli esempi (vedi
    ``ripara_escape_json``). Il JSON valido non viene mai toccato.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"[f1a] JSON non valido dal modello (escape a {exc.lineno}:{exc.colno}).")

    if rigenera is not None:
        secondo = rigenera()
        try:
            parsed = json.loads(secondo)
            print("[f1a] Seconda richiesta al modello: JSON valido, testo integro.")
            return parsed
        except json.JSONDecodeError:
            print("[f1a] Anche la seconda risposta e' rotta: riparo gli escape.")
            raw = secondo

    riparato = ripara_escape_json(raw)
    try:
        return json.loads(riparato)
    except json.JSONDecodeError:
        # La riparazione non e' bastata. Il payload finisce su disco PRIMA di
        # rilanciare: senza, l'unica traccia e' la posizione dell'errore nel log
        # e la risposta del modello e' persa, quindi la correzione successiva si
        # fa a indovinare. Con seed e temperature fissi il caso e' riproducibile,
        # ma solo se il testo che l'ha causato resta.
        cartella = Path(os.getenv("OUTPUT_DIR", ".")).resolve()
        cartella.mkdir(parents=True, exist_ok=True)
        (cartella / "f1a_risposta_non_parsabile.txt").write_text(raw, encoding="utf-8")
        (cartella / "f1a_risposta_riparata.txt").write_text(riparato, encoding="utf-8")
        print(f"[f1a] Payload non parsabile salvato in {cartella}/f1a_risposta_non_parsabile.txt")
        raise


def request_analysis(system_prompt: str, user_prompt: str, config: F1Config) -> dict:
    model = OpenAI(base_url=config.openai_base_url, api_key=config.openai_api_key)
    request = {
        "model": config.model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "timeout": config.llm_timeout_seconds,
        # Nemotron-3 Super ragiona prima di rispondere se non glielo si vieta:
        # qui l'output e' vincolato da un json_schema strict, quindi i token di
        # thinking sono latenza pura. Il campo va passato dentro
        # ``chat_template_kwargs`` — il parametro top-level ``thinking`` viene
        # rifiutato dall'endpoint con 400 extra_forbidden. Stessa forma usata da
        # f0, f1b e f1c.
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}, "top_k": 1},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "golden_set_akn",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "label_rilevanti": {
                            "type": "array",
                            "items": {"type": "string", "enum": sorted(ALLOWED_LABELS)},
                        },
                        "regole_tipografiche": {
                            "type": "object",
                            "properties": {
                                "formato_commi": {"type": "string"},
                                "formato_liste": {"type": "string"},
                            },
                            "required": ["formato_commi", "formato_liste"],
                            "additionalProperties": False,
                        },
                        "golden_set": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "testo_originale": {"type": "string"},
                                    "etichetta": {"type": "string", "enum": sorted(ALLOWED_LABELS)},
                                    "ragionamento": {"type": "string"},
                                    "confidenza_attesa": {"type": "number"},
                                },
                                "required": ["testo_originale", "etichetta", "ragionamento", "confidenza_attesa"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["label_rilevanti", "regole_tipografiche", "golden_set"],
                    "additionalProperties": False,
                },
            },
        },
    }
    response = model.chat.completions.create(**request)

    def rigenera() -> str:
        # Stesso prompt, campionamento diverso: con seed e temperature identici
        # si riotterrebbe la stessa risposta rotta, e il tentativo sarebbe inutile.
        print("[f1a] Rieseguo la richiesta al modello per questo documento.")
        retry = dict(request, seed=request["seed"] + 1, temperature=0.2)
        return model.chat.completions.create(**retry).choices[0].message.content

    partial = parse_llm_json(response.choices[0].message.content, rigenera=rigenera)
    return {
        "mappa_gerarchica": AKN_LEVELS,
        "label_rilevanti": partial["label_rilevanti"],
        "regole_tipografiche": partial["regole_tipografiche"],
        "pattern_classificazione": AKN_PATTERNS,
        "golden_set": partial["golden_set"],
    }


def sample_line_text(line: str) -> str:
    return re.sub(r"^\[page=.*?\]\s*", "", line).strip()


def infer_sample_label(text: str) -> str | None:
    structural = infer_structural_label(text)
    if structural:
        return structural
    normalized = text.strip()
    if PARAGRAPH_MARKER_RE.match(normalized):
        return "paragraph"
    if POINT_MARKER_RE.match(normalized):
        return "point"
    if PAGE_NUMBER_NOISE_RE.fullmatch(normalized):
        return "noise"
    return "content" if normalized else None


def sample_examples_by_label(sample_text: str) -> dict[str, list[str]]:
    examples: dict[str, list[str]] = defaultdict(list)
    for line in sample_text.splitlines():
        if not line or line.startswith("[..."):
            continue
        text = sample_line_text(line)
        label = infer_sample_label(text)
        if label in ALLOWED_LABELS and text and text not in examples[label]:
            examples[label].append(text)
    return examples


def validate_and_save_analysis(result: dict, config: F1Config, sample_text: str, signature: dict) -> dict:
    clean_golden = []
    seen = set()
    for item in result.get("golden_set", []):
        text = (item.get("testo_originale") or "").strip()
        label = (item.get("etichetta") or "").lower().strip()
        if not text or label not in ALLOWED_LABELS:
            continue
        if text not in sample_text:
            continue
        key = (label, text)
        if key in seen:
            continue
        clean_golden.append(
            {
                "testo_originale": text,
                "etichetta": label,
                "ragionamento": item.get("ragionamento") or "Esempio copiato dal campione.",
                "confidenza_attesa": item.get("confidenza_attesa"),
            }
        )
        seen.add(key)
        if len(clean_golden) >= config.n_llm_examples:
            break

    examples_by_label = sample_examples_by_label(sample_text)
    requested_labels = set(result.get("label_rilevanti", [])) & ALLOWED_LABELS
    requested_labels |= {label for label in examples_by_label if label in AKN_LEVELS}
    covered_labels = {item["etichetta"] for item in clean_golden}
    for missing_label in sorted(requested_labels - covered_labels):
        for example in examples_by_label.get(missing_label, []):
            key = (missing_label, example)
            if key in seen:
                continue
            clean_golden.append(
                {
                    "testo_originale": example,
                    "etichetta": missing_label,
                    "ragionamento": "Esempio deterministico presente verbatim nel campione.",
                    "confidenza_attesa": 0.75,
                }
            )
            seen.add(key)
            covered_labels.add(missing_label)
            break

    # La copertura si pretende SOLO per le label che il campione attesta davvero.
    # ``requested_labels`` mescola due cose diverse: i livelli riconosciuti dallo
    # scan deterministico (attestati per costruzione) e le label che il modello
    # ha dichiarato rilevanti, che possono benissimo non esistere in questo
    # documento — un executive order statunitense e' fatto di "Sec. 1." e non ha
    # articoli, ma il modello elenca ``article`` perche' se lo aspetta da un atto
    # normativo. Pretendere un esempio verbatim per una label assente faceva
    # fallire l'intero batch su un documento perfettamente valido.
    attested_labels = {label for label in requested_labels if examples_by_label.get(label)}
    still_missing = sorted(label for label in attested_labels if label not in covered_labels)
    if still_missing:
        raise ValueError(f"Golden set senza copertura minima per label presenti nel campione: {still_missing}")

    if not clean_golden:
        raise ValueError("Golden set vuoto: il LLM non ha restituito esempi validi.")

    # Una label dichiarata rilevante ma senza riscontro nel campione non e' un
    # errore, e' un'aspettativa smentita dal documento: la si toglie dalle label
    # per cui f1b cerchera' esempi simili, dichiarandolo.
    unattested = sorted(requested_labels - attested_labels - covered_labels)
    if unattested:
        print(f"[f1a] Label dichiarate rilevanti ma assenti dal campione, rimosse: {unattested}")

    result["mappa_gerarchica"] = AKN_LEVELS
    result["pattern_classificazione"] = AKN_PATTERNS
    result["label_rilevanti"] = [
        x for x in result.get("label_rilevanti", []) if x in ALLOWED_LABELS and x not in unattested
    ]
    result["label_non_attestate"] = unattested
    result["golden_set"] = clean_golden
    result["cache_signature"] = signature
    write_json(config.analysis_json_path, result)

    counts = defaultdict(int)
    for item in clean_golden:
        counts[item["etichetta"]] += 1
    print("[OK] Analisi salvata:", config.analysis_json_path)
    print("Distribuzione golden set:", dict(counts))
    return result


def save_embedding_db(result: dict, config: F1Config) -> None:
    print("Embedding model:", config.embedding_model_name, "| device:", config.embedding_device)
    model_emb = SentenceTransformer(config.embedding_model_name, device=config.embedding_device)
    texts = [item["testo_originale"] for item in result["golden_set"]]
    labels = [item["etichetta"] for item in result["golden_set"]]
    vectors = model_emb.encode(texts, normalize_embeddings=True).astype("float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    np.savez_compressed(
        config.embedding_db_path,
        embeddings=vectors,
        testi=np.array(texts, dtype=object),
        etichette=np.array(labels, dtype=object),
    )
    print("Embedding DB salvato:", config.embedding_db_path)
    print("Esempi:", len(texts), "| dimensione:", vectors.shape[1])


def release_model_before_embeddings() -> None:
    if os.getenv("STOP_MODEL_BEFORE_EMBEDDINGS", "1") == "1":
        stop_big_model("prima di caricare SentenceTransformer per gli embedding")


def prepare_analysis(config: F1Config | None = None, stage: str | None = None) -> dict:
    """Analisi preliminare del documento, in una o due fasi.

    f1a fa due lavori indipendenti: l'analisi del campione con l'LLM e la
    costruzione del DB di embedding con SentenceTransformer. Su una macchina a
    memoria unificata i due modelli non ci stanno insieme, quindi il secondo
    lavoro richiede che il primo abbia liberato la memoria.

    ``stage`` permette all'orchestratore di eseguirli separatamente su TUTTI i
    documenti: ``"analysis"`` per tutti con il modello acceso, poi un solo
    spegnimento, poi ``"embeddings"`` per tutti. Eseguiti insieme (``"all"``,
    il default, e l'unico comportamento in modalita' per-documento) obbligano
    invece a un ciclo spegni/riaccendi per ogni documento — con un avvio di
    TensorRT-LLM da ~10 minuti, ore buttate.
    """
    config = config or load_config()
    stage = (stage or os.getenv("F1A_STAGE") or "all").strip().lower()
    if stage not in {"all", "analysis", "embeddings"}:
        raise ValueError(f"F1A_STAGE non valido: {stage!r} (attesi: all, analysis, embeddings)")

    if stage == "embeddings":
        # L'analisi e' gia' stata scritta dalla fase precedente: qui non serve
        # ne' il campione ne' l'LLM, solo l'encoder.
        result = load_json(config.analysis_json_path)
        print(f"[f1a] Fase 'embeddings': golden set da {config.analysis_json_path}")
        save_embedding_db(result, config)
        return result

    print("Label AKN ammesse:", sorted(ALLOWED_LABELS))
    blocks = load_blocks(config)
    _, sample_text = build_sample_text(blocks, config)
    system_prompt, user_prompt = build_analysis_prompts(sample_text, config)
    signature = analysis_cache_signature(config, sample_text, system_prompt, user_prompt)

    result = load_cached_analysis(config, signature)
    if result is None:
        result = request_analysis(system_prompt, user_prompt, config)
    result = validate_and_save_analysis(result, config, sample_text, signature)

    if stage == "analysis":
        # Il modello resta acceso: lo spegnera' l'orchestratore una volta sola,
        # dopo aver analizzato tutti i documenti.
        print("[f1a] Fase 'analysis' completata; embedding DB rimandato alla fase dedicata.")
        return result

    release_model_before_embeddings()
    save_embedding_db(result, config)
    return result


def main() -> None:
    prepare_analysis()


if __name__ == "__main__":
    main()
