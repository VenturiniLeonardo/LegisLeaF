from __future__ import annotations

import json
import math
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from legisleaf.common import (
    ALLOWED_LABELS,
    PAGE_NUMBER_NOISE_RE,
    PARAGRAPH_MARKER_RE,
    POINT_MARKER_RE,
    LABEL_DESCRIPTIONS,
    LANGUAGE,
    F1Config,
    allowed_text,
    apply_strict_structural_validation,
    infer_structural_label,
    load_blocks,
    load_config,
    load_json,
    normalize_legal_text,
    tronca,
    write_json,
)

# Confidenza dei blocchi riconosciuti dalle regex strutturali, non dal modello.
# Era 0.86, sotto la soglia di accettazione: ogni riconoscimento deterministico
# veniva mandato alla revisione LLM, cioe' si chiedeva a un modello di rivedere
# un match esatto di pattern. Un'intestazione che supera
# apply_strict_structural_validation e' il caso piu' affidabile che la pipeline
# produca: deve stare sopra qualunque soglia ragionevole.
DETERMINISTIC_STRUCTURAL_CONFIDENCE = 0.97


def label_confidence_from_logprobs(response, label: str) -> float | None:
    """Probabilita' che il modello assegna all'etichetta che ha scelto.

    f1b chiedeva i logprobs al server e poi ritornava ``None`` incondizionatamente:
    la soglia ``CONFIDENCE_ACCEPT_THRESHOLD`` non poteva mai essere superata e
    OGNI blocco passava dalla revisione LLM di f1c. L'architettura a due stadi —
    SLM che decide, LLM che rivede solo l'incerto — era disattivata di fatto.

    Si considerano i soli token che compongono il VALORE dell'etichetta, non
    l'intera risposta: graffe e nome del campo sono imposti dallo schema strict,
    hanno probabilita' prossima a 1 e diluirebbero il segnale.

    Il valore restituito e' la MEDIA GEOMETRICA delle probabilita' per token, non
    il loro prodotto. Il prodotto renderebbe le etichette non confrontabili fra
    loro a soglia fissa: "content" e' un token solo, "unresolved" ne occupa due
    ("un" + "resolved"), e due token al 97% danno 0.94 mentre uno solo al 97%
    da' 0.97. Con una soglia unica il prodotto penalizzerebbe sistematicamente
    le etichette piu' lunghe, che finirebbero in revisione per la loro
    tokenizzazione e non per un'incertezza del modello. La media geometrica e'
    la probabilita' per token, e a soglia fissa confronta cose omogenee.

    Ritorna ``None`` — cioe' il comportamento precedente, revisione LLM — quando
    il backend non restituisce i logprobs o quando i token dell'etichetta non
    sono localizzabili: meglio una revisione in piu' di una confidenza inventata.
    """
    logprobs = getattr(response.choices[0], "logprobs", None)
    tokens = getattr(logprobs, "content", None) if logprobs is not None else None
    if not tokens:
        return None

    # Ricostruisce il testo emesso token per token, tenendo l'intervallo di
    # caratteri coperto da ciascuno: serve a individuare quali token formano
    # l'etichetta, che puo' essere spezzata in piu' pezzi ("un", "resolved").
    testo = ""
    intervalli: list[tuple[int, int, int]] = []
    for indice, token in enumerate(tokens):
        pezzo = getattr(token, "token", "") or ""
        intervalli.append((len(testo), len(testo) + len(pezzo), indice))
        testo += pezzo

    posizione = testo.lower().rfind(label)
    if posizione < 0:
        return None

    fine = posizione + len(label)
    somma = 0.0
    trovati = 0
    for inizio_token, fine_token, indice in intervalli:
        if fine_token <= posizione or inizio_token >= fine:
            continue
        valore = getattr(tokens[indice], "logprob", None)
        if valore is None:
            return None
        somma += valore
        trovati += 1

    if not trovati:
        return None
    return math.exp(somma / trovati)


# Parallelismo delle chiamate allo SLM. f1b interrogava il modello un blocco per
# volta in un ciclo sequenziale: 16.482 richieste in serie sul corpus inglese,
# mentre f1c — che ne fa un ordine di grandezza meno — era gia' parallelizzato.
SLM_MAX_WORKERS = int(os.getenv("SLM_MAX_WORKERS", "4"))

# Salto deterministico esteso ai marcatori di contenuto. Prima riguardava solo le
# intestazioni strutturali: 9.2% dei blocchi. Comma numerato, punto di elenco e
# numero di pagina nudo sono altrettanto deterministici, e portano la quota al
# 57.8%. Misurato sul corpus inglese: l'etichetta della regex coincide con quella
# finale della pipeline nel 93.4% dei casi (95.3% escludendo i blocchi che la
# pipeline lascia 'unresolved'), cioe' quanto la coppia SLM+LLM concorda con se
# stessa. Mettere a 0 per tornare al solo salto strutturale.
DETERMINISTIC_CONTENT_MARKERS = os.getenv("DETERMINISTIC_CONTENT_MARKERS", "1") == "1"


def deterministic_label_for(block: dict, testo: str) -> tuple[str | None, str | None]:
    """Etichetta ricavabile dal solo testo, senza interrogare il modello.

    Ritorna ``(etichetta, motivo)`` oppure ``(None, None)`` quando serve lo SLM.
    I blocchi ``recovered_extra_candidate`` sono esclusi: provengono da un
    recupero incerto dell'estrazione e vanno sempre validati dal modello.
    """
    if block.get("recovered_extra_candidate"):
        return None, None

    structural = infer_structural_label(testo)
    if structural:
        return structural, "deterministic_structural_header"

    if not DETERMINISTIC_CONTENT_MARKERS:
        return None, None

    normalizzato = normalize_legal_text(testo)
    if PAGE_NUMBER_NOISE_RE.fullmatch(normalizzato):
        return "noise", "deterministic_page_number"
    if PARAGRAPH_MARKER_RE.match(normalizzato):
        return "paragraph", "deterministic_paragraph_marker"
    if POINT_MARKER_RE.match(normalizzato):
        return "point", "deterministic_point_marker"
    return None, None


def load_embedding_index(config: F1Config):
    if not config.embedding_db_path.exists():
        raise FileNotFoundError(f"Embedding DB mancante: {config.embedding_db_path}. Esegui prima f1a_prepare_analysis.")
    data = np.load(config.embedding_db_path, allow_pickle=True)
    vectors = data["embeddings"].astype("float32")
    texts = data["testi"].tolist()
    labels = data["etichette"].tolist()
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    embeddings = [
        {"testo_originale": text, "etichetta": label, "embedding": vector}
        for text, label, vector in zip(texts, labels, vectors)
    ]
    return index, embeddings


def esempi_simili(query_vector: np.ndarray, index, embeddings: list[dict], config: F1Config) -> list[dict]:
    """Cerca esempi simili usando un vettore embedding già pre-calcolato."""
    scores, positions = index.search(query_vector, min(config.n_similar_examples, index.ntotal))
    found = []
    for score, pos in zip(scores[0], positions[0]):
        if pos < 0:
            continue
        found.append(
            {
                "testo_originale": embeddings[pos]["testo_originale"],
                "etichetta": embeddings[pos]["etichetta"],
                "score": float(score),
            }
        )
    return found


def build_slm_system_prompt(result: dict) -> str:
    mappa_gerarchica_str = "\n".join(
        f"- {nome} (livello {livello})"
        for nome, livello in sorted(result.get("mappa_gerarchica", {}).items(), key=lambda x: x[1])
    )
    regole_tipografiche = result.get("regole_tipografiche", {})
    etichette_str = "\n".join(f"- {label}: {LABEL_DESCRIPTIONS[label]}" for label in sorted(ALLOWED_LABELS))

    return f"""Sei un classificatore automatico di testi giuridici in lingua {LANGUAGE.name}, specializzato in struttura normativa Akoma Ntoso.

Il tuo compito e' assegnare UN'UNICA etichetta strutturale al BLOCCO CORRENTE,
scegliendola ESCLUSIVAMENTE tra quelle elencate di seguito.

STRUTTURA GERARCHICA DEL DOCUMENTO:
{mappa_gerarchica_str}

REGOLE TIPOGRAFICHE OSSERVATE NEL DOCUMENTO:
- Formato commi/paragraph: {regole_tipografiche.get('formato_commi', 'non specificato')}
- Formato liste/point: {regole_tipografiche.get('formato_liste', 'non specificato')}

ETICHETTE VALIDE (SOLO queste, nessun'altra):
{etichette_str}

NOTE:
- Riceverai blocco precedente, blocco corrente e blocco successivo: usa precedente e successivo solo come contesto.
- Riceverai anche la label del blocco precedente quando gia' disponibile: usala come indizio di continuita' gerarchica, non come vincolo assoluto.
- Devi classificare SOLO il blocco corrente.
- Ignora come possibili label i tipi visuali dell'estrattore: section_header, list_item, text, article_header, title, plain_text, picture.
- Usa i tipi visuali solo come indizi, mai come etichette finali.
- I livelli gerarchici si riconoscono dalle regole tipografiche e dalla struttura del documento descritte sopra.
{LANGUAGE.hierarchy_glosses_text(indent="", suffix=".", exclude=("article",))}
- article: usa questa label solo per intestazioni autonome come {LANGUAGE.examples_text('article')}; non usare article per paragraph che citano o contengono un articolo.
- paragraph: segue il formato commi descritto sopra, inclusi inizi come {LANGUAGE.examples_text('paragraph')} quando sono contenuto intra-articolare.
- point: segue il formato liste descritto sopra, per esempio {LANGUAGE.examples_text('point')}.
- content: testo discorsivo non numerato, preamboli, paragrafi, citazioni normative o testo normativo ordinario.
- noise: elementi non normativi, intestazioni, date, titoli editoriali, codici alfanumerici, numeri di pagina o rumore OCR.
- Non usare sinonimi in lingua {LANGUAGE.name} come {LANGUAGE.forbidden_synonyms_text()}.

Rispondi ESCLUSIVAMENTE con un JSON valido nel formato:
{{"etichetta": "<nome_etichetta>"}}

Non aggiungere testo, spiegazioni o markdown."""


def block_metadata_for_prompt(block: dict) -> str:
    metadata = {
        "bbox": block.get("bbox"),
        "bbox_inherited": block.get("bbox_inherited"),
        "document_zone": block.get("document_zone"),
        "block_type_original": block.get("block_type_original"),
        "block_type_canonical": block.get("block_type_canonical", block.get("block_type")),
        "ocr_score": block.get("ocr_score"),
        "selected_model": block.get("selected_model", block.get("selected_from")),
        "source_model": block.get("source_model"),
        "page_agreement_score": block.get("page_agreement_score"),
        "recovered_extra_candidate": block.get("recovered_extra_candidate"),
    }
    return json.dumps(metadata, ensure_ascii=False, sort_keys=True)


def source_metadata(block: dict) -> dict:
    return {
        "selected_from": block.get("selected_from"),
        "selected_model": block.get("selected_model"),
        "source_model": block.get("source_model"),
        "block_type": block.get("block_type"),
        "block_type_original": block.get("block_type_original"),
        "block_type_canonical": block.get("block_type_canonical"),
        "document_zone": block.get("document_zone"),
        "bbox": block.get("bbox"),
        "bbox_inherited": block.get("bbox_inherited"),
        "ocr_score": block.get("ocr_score"),
        "recovered_extra_candidate": block.get("recovered_extra_candidate"),
        "page_consensus_accuracy": block.get("page_consensus_accuracy"),
        "page_agreement_score": block.get("page_agreement_score"),
        "source_order": block.get("source_order"),
    }


def prompt_classificazione(
    testo: str,
    precedente: str,
    successivo: str,
    label_precedente: str,
    esempi: list[dict],
    config: F1Config,
    block: dict | None = None,
) -> str:
    esempi_txt = "\n".join(
        f"- label={e['etichetta']} | score={e['score']:.3f} | testo={tronca(e['testo_originale'], 350)}"
        for e in esempi
    ) or "Nessun esempio disponibile."

    return f"""Classifica SOLO il blocco corrente usando Akoma Ntoso.

Etichette ammesse, senza eccezioni:
{allowed_text()}

Regole:
- Rispondi solo con JSON valido.
- Usa solo una delle etichette ammesse.
- Non usare label OCR come text, title, section_header, list_item, article_header, plain_text, picture.
- Se il testo e' una istruzione modificativa ({LANGUAGE.modificative_examples_text()}), classificalo content: non e' una intestazione.
- annex e' una intestazione autonoma di allegato, per esempio {LANGUAGE.examples_text('annex')}.
- title/chapter/section accettano numeri romani o arabi, per esempio {LANGUAGE.examples_text('title')}, {LANGUAGE.examples_text('chapter')}, {LANGUAGE.examples_text('section')}.
- article e' solo una intestazione autonoma tipo {LANGUAGE.examples_text('article')}.
- paragraph e' un comma numerato: {LANGUAGE.examples_text('paragraph')}, ecc.
- point e' una voce di elenco: {LANGUAGE.examples_text('point')}, ecc.
- content e' testo normativo ordinario.
- noise e' testo editoriale, pagina, intestazione, pubblicita', errore OCR.

Contesto:
PRECEDENTE ({label_precedente}): {tronca(precedente, config.max_chars_testo)}
CORRENTE: {tronca(testo, config.max_chars_testo)}
SUCCESSIVO: {tronca(successivo, config.max_chars_testo)}
METADATI_CORRENTE: {block_metadata_for_prompt(block or {})}

Esempi simili:
{esempi_txt}

Formato obbligatorio:
{{"etichetta": "<una label ammessa>"}}
"""


def call_slm(prompt: str, slm_system_prompt: str, client: OpenAI, config: F1Config) -> dict:
    request = dict(
        model=config.model_name_local,
        messages=[
            {"role": "system", "content": slm_system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        top_p=1.0,
        # La risposta è sempre {"etichetta": "<label>"}: bastano ~20 token.
        # 32 aggiunge margine sufficiente senza sprecare budget di generazione.
        max_tokens=32,
        seed=42,
        # Disabilita il thinking su Qwen3/vLLM e forza decoding greedy.
        extra_body={"chat_template_kwargs": {"enable_thinking": False}, "top_k": 1},
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "classificazione_akn",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"etichetta": {"type": "string", "enum": sorted(ALLOWED_LABELS)}},
                    "required": ["etichetta"],
                    "additionalProperties": False,
                },
            },
        },
    )

    use_logprobs = config.enable_logprobs
    if use_logprobs:
        request["logprobs"] = True
        request["top_logprobs"] = 5

    try:
        response = client.chat.completions.create(**request)
    except Exception as exc:
        msg = str(exc).lower()
        if use_logprobs and ("logprob" in msg or "unsupported" in msg or "unexpected" in msg):
            warnings.warn("Logprobs non supportati dal backend SLM: continuo senza confidence.")
            request.pop("logprobs", None)
            request.pop("top_logprobs", None)
            use_logprobs = False
            response = client.chat.completions.create(**request)
        else:
            raise

    raw = (response.choices[0].message.content or "").strip()
    parsed = json.loads(raw)
    label = (parsed.get("etichetta") or "").lower().strip()
    if label not in ALLOWED_LABELS:
        raise ValueError(f"Etichetta non ammessa: {label}")

    confidence = label_confidence_from_logprobs(response, label) if use_logprobs else None
    return {"etichetta": label, "label_confidence": confidence, "confidence": confidence}


def classify_blocks_slm(config: F1Config | None = None) -> dict[str, list[dict]]:
    config = config or load_config()
    if not config.analysis_json_path.exists():
        raise FileNotFoundError(f"Analisi mancante: {config.analysis_json_path}. Esegui prima f1a_prepare_analysis.")

    result = load_json(config.analysis_json_path)
    blocks = load_blocks(config)
    index, embeddings = load_embedding_index(config)
    print("Embedding model:", config.embedding_model_name, "| device:", config.embedding_device)
    model_emb = SentenceTransformer(config.embedding_model_name, device=config.embedding_device)
    model_local = OpenAI(base_url=config.openai_base_url, api_key=config.openai_api_key)
    slm_system_prompt = build_slm_system_prompt(result)

    risultati_slm = []
    risultati = []
    bassa_confidenza = []
    revisioni_llm = []
    fallimenti = []

    print("Blocchi da classificare:", len(blocks))
    print("SLM:", config.model_name_local)

    # Pre-calcola tutti gli embedding in un unico batch per evitare N encode() singoli.
    # model_emb.encode con batch_size sfrutta la GPU/CPU in modo vettorializzato.
    print("Pre-calcolo embedding in batch...")
    testi_blocks = [b["testo"] for b in blocks]
    all_embeddings = model_emb.encode(
        testi_blocks,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True,
    ).astype("float32")
    print("Embedding completati.")

    # --- Fase 1: etichette deterministiche, senza interrogare il modello --- #
    # Va per prima anche perche' popola ``etichetta_predetta``, che la fase 2 usa
    # come indizio di continuita' gerarchica per i blocchi che seguono.
    record_per_indice: dict[int, dict] = {}
    da_classificare: list[int] = []

    for i, block in enumerate(blocks):
        testo = block["testo"]
        etichetta, motivo = deterministic_label_for(block, testo)
        if not etichetta:
            da_classificare.append(i)
            continue

        block["etichetta_predetta"] = etichetta
        record_per_indice[i] = {
            "file": block["file"],
            "ordine": block["ordine"],
            "page": block["page"],
            "page_order": block["page_order"],
            "testo": testo,
            "testo_preview": testo[:160],
            "testo_normalizzato": normalize_legal_text(testo),
            "testo_precedente": blocks[i - 1]["testo"] if i > 0 else "",
            "testo_successivo": blocks[i + 1]["testo"] if i + 1 < len(blocks) else "",
            "etichetta": etichetta,
            "candidate_label": etichetta,
            "deterministic_evidence_score": DETERMINISTIC_STRUCTURAL_CONFIDENCE,
            "label_confidence": DETERMINISTIC_STRUCTURAL_CONFIDENCE,
            "confidence": DETERMINISTIC_STRUCTURAL_CONFIDENCE,
            "esempi_simili": [],
            "raw": {
                "candidate_label": etichetta,
                "source": motivo,
                "requires_context_validation": True,
            },
            "review_status": "deterministic",
            "source": source_metadata(block),
        }

    print(
        f"Deterministici: {len(record_per_indice)}/{len(blocks)} "
        f"({len(record_per_indice) / max(len(blocks), 1):.1%}); "
        f"chiamate allo SLM: {len(da_classificare)} con {SLM_MAX_WORKERS} worker"
    )

    # --- Fase 2: chiamate allo SLM, in parallelo ---------------------------- #
    def classifica(i: int) -> tuple[int, dict, dict | None]:
        """Classifica un blocco. Nessuno stato condiviso: solo letture."""
        block = blocks[i]
        testo = block["testo"]
        precedente = blocks[i - 1]["testo"] if i > 0 else ""
        successivo = blocks[i + 1]["testo"] if i + 1 < len(blocks) else ""
        # Disponibile quando il blocco precedente e' stato risolto in fase 1;
        # vuoto quando anche quello attende lo SLM. E' il prezzo dichiarato del
        # parallelismo: l'indizio resta dove e' piu' informativo (dopo una
        # intestazione riconosciuta) e cade dove sarebbe stato a sua volta una
        # predizione. Con SLM_MAX_WORKERS=1 il comportamento non cambia comunque,
        # perche' l'ordine di esecuzione resta quello dei blocchi.
        label_precedente = blocks[i - 1].get("etichetta_predetta", "") if i > 0 else ""

        esempi = esempi_simili(all_embeddings[i : i + 1], index, embeddings, config)
        prompt = prompt_classificazione(testo, precedente, successivo, label_precedente, esempi, config, block)

        ultimo_errore: Exception | None = None
        for tentativo in range(1, config.max_retries + 1):
            try:
                risposta = call_slm(prompt, slm_system_prompt, model_local, config)
                label = risposta["etichetta"]
                source = source_metadata(block)
                label, strict_override = apply_strict_structural_validation(label, testo, source)
                block["etichetta_predetta"] = label

                record = {
                    "file": block["file"],
                    "ordine": block["ordine"],
                    "page": block["page"],
                    "page_order": block["page_order"],
                    "testo": testo,
                    "testo_preview": testo[:160],
                    "testo_precedente": precedente,
                    "testo_successivo": successivo,
                    "etichetta": label,
                    "label_confidence": risposta.get("label_confidence"),
                    "confidence": risposta.get("confidence"),
                    "esempi_simili": esempi,
                    "raw": risposta,
                    "source": source,
                }
                if strict_override:
                    record["strict_validation"] = strict_override

                conf = record["label_confidence"]
                accettato = isinstance(conf, (int, float)) and conf > config.confidence_accept_threshold
                record["review_status"] = "accepted_slm" if accettato else "pending_llm_review"
                return i, record, None
            except Exception as exc:
                ultimo_errore = exc
                if tentativo < config.max_retries:
                    tqdm.write(f"Tentativo {tentativo} fallito (blocco {block['ordine']}): {exc}")

        fallito = {
            "file": block["file"],
            "ordine": block["ordine"],
            "page": block["page"],
            "page_order": block["page_order"],
            "testo": testo,
            "testo_preview": testo[:160],
            "testo_precedente": precedente,
            "testo_successivo": successivo,
            "etichetta": "unresolved",
            "label_confidence": None,
            "confidence": None,
            "esempi_simili": esempi,
            "raw": {"error": str(ultimo_errore)},
            "review_status": "slm_failed_unresolved",
            "source": source_metadata(block),
        }
        return i, fallito, {
            "file": block["file"],
            "ordine": block["ordine"],
            "page": block["page"],
            "testo": testo[:300],
            "errore": str(ultimo_errore),
            "fallback_label": "unresolved",
        }

    if da_classificare:
        with ThreadPoolExecutor(max_workers=max(1, SLM_MAX_WORKERS)) as pool:
            iteratore = pool.map(classifica, da_classificare)
            for i, record, fallimento in tqdm(
                iteratore, total=len(da_classificare), desc="Classificazione SLM", unit="blocco"
            ):
                record_per_indice[i] = record
                if fallimento is not None:
                    fallimenti.append(fallimento)

    # --- Assemblaggio in ordine di blocco ---------------------------------- #
    # I risultati vanno scritti nell'ordine dei blocchi, non in quello di
    # completamento dei thread: ``review_id`` indicizza la lista che f1c rilegge,
    # e un ordine non deterministico renderebbe due run non confrontabili.
    for i in range(len(blocks)):
        record = record_per_indice.get(i)
        if record is None:
            continue
        risultati_slm.append(record)
        if record["review_status"] == "pending_llm_review":
            record["review_id"] = len(bassa_confidenza)
            bassa_confidenza.append(record)
        else:
            risultati.append(record)

    write_json(config.classification_results_path, risultati)
    write_json(config.classification_slm_raw_path, risultati_slm)
    write_json(config.low_confidence_results_path, bassa_confidenza)
    write_json(config.llm_reviewed_results_path, revisioni_llm)
    write_json(config.classification_failures_path, fallimenti)

    print("Classificati dallo SLM:", len(risultati_slm))
    print("Accettati subito:", len(risultati))
    print("Da revisionare con LLM:", len(bassa_confidenza))
    print("Fallimenti:", len(fallimenti))
    print("File intermedi salvati.")
    return {
        "risultati_slm": risultati_slm,
        "risultati": risultati,
        "bassa_confidenza": bassa_confidenza,
        "revisioni_llm": revisioni_llm,
        "fallimenti": fallimenti,
    }


def main() -> None:
    classify_blocks_slm()


if __name__ == "__main__":
    main()
