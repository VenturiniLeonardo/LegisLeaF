from __future__ import annotations

import json
import os
import re
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm.auto import tqdm

from legisleaf.common import (
    ALLOWED_LABELS,
    LANGUAGE,
    F1Config,
    allowed_text,
    apply_strict_structural_validation,
    load_config,
    load_json,
    tronca,
    write_json,
)


def accept_review_label(item: dict, label: str) -> tuple[str, dict | None]:
    """Etichetta finale per un blocco rivisto dall'LLM, dopo il filtro strutturale.

    Il verdetto dell'LLM passa per lo stesso controllo deterministico applicato in
    f1b alle predizioni dell'SLM: un'etichetta strutturale viene accettata solo se
    il testo ha davvero la forma di un'intestazione. Senza questo filtro l'LLM
    promuoveva ad ``article`` le rubriche ("Definizioni", "Sanzioni", "Oggetto"),
    creando nodi-articolo fantasma privi di numero."""
    return apply_strict_structural_validation(label, item.get("testo") or "", item)


def cached_review_verdicts(config: F1Config) -> dict[tuple, dict]:
    """Verdetti LLM gia' salvati da una revisione precedente, per identita' di blocco.

    Serve alla modalita' ``F1C_REUSE_CACHED_REVIEW=1``: ri-deriva la classificazione
    finale dai verdetti gia' ottenuti, senza richiamare il modello. Utile quando
    cambia solo la logica DETERMINISTICA a valle del verdetto (es. il filtro
    strutturale): rieseguire l'LLM introdurrebbe variazioni proprie, rendendo
    impossibile attribuire la differenza alla modifica del codice."""
    if not config.llm_reviewed_results_path.exists():
        return {}
    verdicts: dict[tuple, dict] = {}
    for record in load_json(config.llm_reviewed_results_path):
        review = record.get("llm_review")
        if review and record.get("review_status") == "llm_reviewed":
            verdicts[record_identity(record)] = review
    return verdicts


def reviewed_record(item: dict, revisione: dict | None) -> dict:
    """Record finale di un blocco a bassa confidenza dato il verdetto LLM (o la sua assenza).

    Usato sia dal percorso live sia dal riuso della cache: e' l'unico punto in cui
    il verdetto viene applicato (filtro strutturale incluso)."""
    fixed = dict(item)
    fixed["etichetta_slm"] = item.get("etichetta")
    fixed["slm_label_confidence"] = item.get("label_confidence")
    fixed["review_confidence"] = None
    fixed["structural_confidence"] = None
    fixed["label_confidence"] = None
    fixed["confidence"] = None
    if revisione:
        label, demotion = accept_review_label(item, revisione["etichetta"])
        fixed["etichetta"] = label
        fixed["llm_review"] = revisione
        fixed["review_status"] = "llm_reviewed"
        if demotion:
            fixed["structural_validation"] = demotion
    else:
        fixed["etichetta"] = "unresolved"
        fixed["review_status"] = "llm_review_missing"
    return fixed


def apply_cached_reviews(bassa_confidenza: list[dict], verdicts: dict[tuple, dict]) -> list[dict]:
    """Ricostruisce i record rivisti dai verdetti in cache, applicando il filtro strutturale."""
    return [reviewed_record(item, verdicts.get(record_identity(item))) for item in bassa_confidenza]


def record_sort_key(record: dict):
    page = record.get("page")
    page_order = record.get("page_order")
    ordine = record.get("ordine")
    return (
        page is None,
        page if page is not None else 0,
        page_order is None,
        page_order if page_order is not None else ordine if ordine is not None else 0,
        ordine if ordine is not None else 0,
    )


def record_identity(record: dict) -> tuple:
    return (
        record.get("file"),
        record.get("ordine"),
        record.get("page"),
        record.get("page_order"),
        record.get("testo"),
    )


def merge_unique_records(records: list[dict]) -> list[dict]:
    priority = {
        "llm_reviewed": 5,
        "accepted_slm": 4,
        "llm_json_error": 3,
        "llm_review_missing": 2,
        "pending_llm_review": 1,
    }
    merged = {}
    for record in records:
        key = record_identity(record)
        old = merged.get(key)
        if old is None or priority.get(record.get("review_status"), 0) >= priority.get(old.get("review_status"), 0):
            merged[key] = record
    return sorted(merged.values(), key=record_sort_key)


def prompt_revisione(batch: list[dict], config: F1Config) -> str:
    items = []
    for item in batch:
        source = item.get("source") or {}
        items.append(
            {
                "id": item["review_id"],
                "testo_precedente": tronca(item.get("testo_precedente"), config.max_chars_testo),
                "testo_corrente": tronca(item.get("testo"), config.max_chars_testo),
                "testo_successivo": tronca(item.get("testo_successivo"), config.max_chars_testo),
                "etichetta_slm": item.get("etichetta"),
                "confidence_slm": item.get("label_confidence"),
                "bbox": source.get("bbox"),
                "document_zone": source.get("document_zone"),
                "block_type_canonical": source.get("block_type_canonical", source.get("block_type")),
                "ocr_score": source.get("ocr_score"),
                "recovered_extra_candidate": source.get("recovered_extra_candidate"),
                "deterministic_evidence": item.get("deterministic_evidence_score"),
            }
        )

    return f"""Revisiona blocchi a bassa confidenza.
Classifica SOLO testo_corrente.
Usa esclusivamente queste etichette:
{allowed_text()}

Regole:
- non inventare label;
- annex per intestazione autonoma di allegato, per esempio {LANGUAGE.examples_text('annex')};
- title/chapter/section accettano numeri romani o arabi, per esempio {LANGUAGE.examples_text('title')}, {LANGUAGE.examples_text('chapter')}, {LANGUAGE.examples_text('section')};
- article solo per intestazione autonoma, per esempio {LANGUAGE.examples_text('article', limit=4)}, anche con titolo dopo il numero;
- le istruzioni modificative ({LANGUAGE.modificative_examples_text()}) restano content;
- paragraph per comma numerato;
- point per elenco;
- content per testo ordinario;
- noise per elementi non normativi.

Rispondi solo con:
{{"revisioni": [{{"id": 0, "etichetta": "content"}}]}}

Blocchi:
{json.dumps(items, ensure_ascii=False, indent=2)}
"""


def parse_first_json(raw: str):
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("{")
    if start < 0:
        raise json.JSONDecodeError("Nessun JSON trovato", raw, 0)
    parsed, end = json.JSONDecoder().raw_decode(raw[start:])
    trailing = raw[start + end :].strip()
    return parsed, trailing


def call_llm_review(prompt: str, client: OpenAI, config: F1Config):
    # Token massimi stimati: ~40 token per elemento (id + etichetta) + overhead JSON.
    max_tokens = config.llm_review_batch_size * 40 + 32
    request = dict(
        model=config.model_name,
        messages=[
            {"role": "system", "content": "Sei un revisore Akoma Ntoso. Rispondi solo con JSON valido."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        top_p=1.0,
        seed=42,
        max_tokens=max_tokens,
        timeout=config.llm_timeout_seconds,
        # Disabilita il thinking di Nemotron-3 Super e forza il decoding greedy
        # (top_k=1) per ridurre latenza e token di output. Il campo va passato
        # come ``chat_template_kwargs.enable_thinking`` (stesso meccanismo di
        # data_eval/generate_golden_queries.py): il parametro top-level
        # ``thinking`` viene invece rifiutato dall'endpoint (400 extra_forbidden),
        # lasciando i blocchi non revisionati -> ``unresolved``.
        extra_body={"chat_template_kwargs": {"enable_thinking": False}, "top_k": 1},
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "revisione_akn",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "revisioni": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "integer"},
                                    "etichetta": {"type": "string", "enum": sorted(ALLOWED_LABELS)},
                                },
                                "required": ["id", "etichetta"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["revisioni"],
                    "additionalProperties": False,
                },
            },
        },
    )

    return client.chat.completions.create(**request)


def parse_review_response(raw: str):
    parsed, trailing = parse_first_json(raw)
    by_id = {}
    for item in parsed.get("revisioni", []):
        label = (item.get("etichetta") or "").lower().strip()
        if label in ALLOWED_LABELS:
            # Normalizza: salva solo id ed etichetta (ragionamento rimosso dallo schema)
            by_id[int(item["id"])] = {"id": int(item["id"]), "etichetta": label}
    return by_id, trailing


MAX_TENTATIVI_MANCANTI = int(os.getenv("F1C_MISSING_RETRIES", "2"))


def review_batch_with_fallback(batch: list[dict], client: OpenAI, config: F1Config, batch_start: int):
    raw_errors = []

    def _review_once(items: list[dict], trace_start: int):
        ids = [item.get("review_id") for item in items]
        if len(ids) != len(set(ids)):
            raise ValueError(f"ID revisione duplicati nel batch: {ids}")
        response = call_llm_review(prompt_revisione(items, config), client, config)
        raw = response.choices[0].message.content or ""
        by_id, trailing = parse_review_response(raw)
        if trailing:
            raw_errors.append({"batch_start": trace_start, "warning": "testo dopo il primo JSON", "preview": trailing[:500]})
        return by_id

    def _recupera_mancanti(items: list[dict], by_id: dict, trace_start: int) -> dict:
        """Richiede i verdetti che il modello non ha restituito.

        Il batch chiede N blocchi in una sola risposta, e il modello puo'
        restituirne meno senza che nulla fallisca: il JSON e' valido, il parsing
        riesce, e i blocchi rimasti fuori diventano 'unresolved' in silenzio. Il
        fallback esistente reagisce solo alle ECCEZIONI, quindi questo caso gli
        passava sotto (24 blocchi su 1966 nella run sul corpus di paper, con
        llm_review_raw_errors.json vuoto: nessuna traccia).

        Si richiedono SOLO i mancanti. Se un tentativo non porta alcun progresso
        si scende a un blocco per richiesta: una risposta con un solo elemento e'
        la forma piu' facile da rispettare per il modello. Quel che resta fuori
        finisce in raw_errors, cosi' smette di essere invisibile.
        """
        for tentativo in range(1, MAX_TENTATIVI_MANCANTI + 1):
            mancanti = [item for item in items if item["review_id"] not in by_id]
            if not mancanti:
                return by_id

            uno_alla_volta = tentativo > 1 or len(mancanti) == 1
            print(
                f"[f1c] {len(mancanti)} verdetti mancanti dal batch {trace_start}: "
                f"tentativo {tentativo}/{MAX_TENTATIVI_MANCANTI}"
                f"{' (uno per richiesta)' if uno_alla_volta else ''}.",
                flush=True,
            )

            gruppi = [[item] for item in mancanti] if uno_alla_volta else [mancanti]
            progresso = False
            for gruppo in gruppi:
                try:
                    nuovi = _review_once(gruppo, trace_start)
                except (json.JSONDecodeError, ValueError) as exc:
                    raw_errors.append({"batch_start": trace_start, "error": str(exc), "retry": "missing_verdicts"})
                    continue
                attesi = {item["review_id"] for item in gruppo}
                for review_id, verdetto in nuovi.items():
                    if review_id in attesi and review_id not in by_id:
                        by_id[review_id] = verdetto
                        progresso = True

            if not progresso and uno_alla_volta:
                break

        ancora_mancanti = [item["review_id"] for item in items if item["review_id"] not in by_id]
        if ancora_mancanti:
            raw_errors.append(
                {
                    "batch_start": trace_start,
                    "error": f"nessun verdetto dal modello per {len(ancora_mancanti)} blocchi",
                    "review_ids": ancora_mancanti,
                    "retry": "missing_verdicts_exhausted",
                }
            )
        return by_id

    try:
        return _recupera_mancanti(batch, _review_once(batch, batch_start), batch_start), raw_errors
    except (json.JSONDecodeError, ValueError) as exc:
        raw_errors.append({"batch_start": batch_start, "error": str(exc), "retry": "split_batch"})

    if len(batch) > 1:
        merged = {}
        mid = len(batch) // 2
        for offset, part in ((0, batch[:mid]), (mid, batch[mid:])):
            try:
                by_id, part_errors = review_batch_with_fallback(part, client, config, batch_start + offset)
                merged.update(by_id)
                raw_errors.extend(part_errors)
            except Exception as exc:
                raw_errors.append({"batch_start": batch_start + offset, "error": str(exc), "retry": "split_failed"})
        return merged, raw_errors

    try:
        return _recupera_mancanti(batch, _review_once(batch, batch_start), batch_start), raw_errors
    except Exception as exc:
        raw_errors.append({"batch_start": batch_start, "error": str(exc), "retry": "individual_failed"})
        return {}, raw_errors


def review_low_confidence(config: F1Config | None = None) -> list[dict]:
    config = config or load_config()
    if not config.analysis_json_path.exists():
        raise FileNotFoundError(f"Analisi mancante: {config.analysis_json_path}. Esegui prima f1a_prepare_analysis.")

    risultati = load_json(config.classification_results_path)
    bassa_confidenza = load_json(config.low_confidence_results_path)
    revisioni_llm = []

    reuse_cached = os.getenv("F1C_REUSE_CACHED_REVIEW", "0") == "1"
    cached = cached_review_verdicts(config) if reuse_cached else {}
    if reuse_cached and not cached:
        raise FileNotFoundError(
            f"F1C_REUSE_CACHED_REVIEW=1 ma non ci sono verdetti riusabili in "
            f"{config.llm_reviewed_results_path}. Esegui prima una revisione LLM completa."
        )

    if cached:
        print(f"Riuso {len(cached)} verdetti LLM in cache (nessuna chiamata al modello).")
        revisioni_llm = apply_cached_reviews(bassa_confidenza, cached)
    elif config.process_low_conf_with_llm:
        client = OpenAI(base_url=config.openai_base_url, api_key=config.openai_api_key)
        raw_errors: list[dict] = []

        # Suddividi in batch e invia in parallelo per saturare il throughput di NIM.
        batches = [
            (start, bassa_confidenza[start : start + config.llm_review_batch_size])
            for start in range(0, len(bassa_confidenza), config.llm_review_batch_size)
        ]
        # results_map[start] = (by_id, batch_errors)
        results_map: dict[int, tuple[dict, list]] = {}

        with ThreadPoolExecutor(max_workers=config.llm_review_max_workers) as executor:
            future_to_start = {
                executor.submit(review_batch_with_fallback, batch, client, config, start): start
                for start, batch in batches
            }
            for future in tqdm(
                as_completed(future_to_start),
                total=len(future_to_start),
                desc="Revisione LLM",
                unit="batch",
            ):
                start = future_to_start[future]
                try:
                    by_id, batch_errors = future.result()
                except Exception as exc:
                    batch_errors = [{"batch_start": start, "error": str(exc), "retry": "executor_failed"}]
                    by_id = {}
                results_map[start] = (by_id, batch_errors)

        # Ricostruisci i risultati nell'ordine originale dei batch.
        for start, batch in batches:
            by_id, batch_errors = results_map[start]
            raw_errors.extend(batch_errors)

            for item in batch:
                revisioni_llm.append(reviewed_record(item, by_id.get(item["review_id"])))

        if raw_errors:
            write_json(config.llm_review_raw_errors_path, raw_errors)
            warnings.warn(f"Problemi parsing revisione LLM: {len(raw_errors)} batch.")
    else:
        for item in bassa_confidenza:
            fixed = dict(item)
            fixed["etichetta_slm"] = item.get("etichetta")
            fixed["slm_label_confidence"] = item.get("label_confidence")
            fixed["etichetta"] = "unresolved"
            fixed["label_confidence"] = None
            fixed["confidence"] = None
            fixed["review_confidence"] = None
            fixed["structural_confidence"] = None
            fixed["review_status"] = "pending_llm_review"
            revisioni_llm.append(fixed)

    risultati_finali = merge_unique_records(risultati + revisioni_llm)

    write_json(config.llm_reviewed_results_path, revisioni_llm)
    write_json(config.classification_results_path, risultati_finali)

    distribuzione = defaultdict(int)
    for item in risultati_finali:
        distribuzione[item.get("etichetta", "noise")] += 1

    print("Revisioni LLM:", len(revisioni_llm))
    # I blocchi senza verdetto restano 'unresolved' e f2 li esclude dalla
    # struttura: vanno detti a voce alta, non lasciati dedurre dalla
    # distribuzione delle etichette.
    senza_verdetto = sum(1 for r in revisioni_llm if r.get("review_status") == "llm_review_missing")
    if senza_verdetto:
        warnings.warn(f"{senza_verdetto} blocchi restano senza verdetto LLM dopo i tentativi di recupero.")
        print(f"ATTENZIONE: {senza_verdetto} blocchi senza verdetto LLM (etichetta 'unresolved').")
    else:
        print("Tutti i blocchi a bassa confidenza hanno ricevuto un verdetto.")
    print("Risultati finali:", len(risultati_finali))
    print("Distribuzione finale:")
    for label, count in sorted(distribuzione.items(), key=lambda x: x[1], reverse=True):
        print(f"- {label}: {count}")
    print("Completato. File finale:", config.classification_results_path)
    return risultati_finali


def main() -> None:
    review_low_confidence()


if __name__ == "__main__":
    main()
