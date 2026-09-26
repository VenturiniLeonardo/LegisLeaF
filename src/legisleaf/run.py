from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from legisleaf.model_manager import ensure_model_running, stop_model, stop_model_before_tree, use_model
from legisleaf.language import baseline_digest, payload_digest
from legisleaf.settings import (
    DEFAULT_OUTPUT_NAMES,
    document_key,
    document_output_dir,
    safe_resource_name,
)


STEP_MODULES = {
    "language": "legisleaf.f0.detect_language",
    "analysis": "legisleaf.f1.prepare_analysis",
    "classification": "legisleaf.f1.classify_blocks_slm",
    "review": "legisleaf.f1.review_low_conf_llm",
    "tree": "legisleaf.f2.build_tree",
}
DEFAULT_STEPS = ["language", "analysis", "classification", "review", "tree"]
STEP_ALIASES = {
    "all": DEFAULT_STEPS,
    "preprocessing": ["language", "analysis"],
    "f0": ["language"],
    "f1": ["analysis", "classification", "review"],
    "f1a": ["analysis"],
    "f1b": ["classification"],
    "f1c": ["review"],
    "f2": ["tree"],
}
MODEL_STEPS = {"language": "big", "analysis": "big", "classification": "small", "review": "big"}
# Step che devono convivere con un server di modello acceso E caricare un encoder
# locale. Oggi solo f1b: classifica via HTTP ma recupera gli esempi few-shot con
# SentenceTransformer. Su memoria unificata il server ha gia' prenotato la GPU
# (kv cache), e l'encoder muore con un CUDA out of memory anche se pesa poche
# centinaia di MB. Gli altri due step che usano un encoder non hanno il problema:
# f1a lo carica in una fase separata a modello spento (vedi step_stages). Per f1b l'encoder serve solo a
# cercare i vicini fra poche decine di esempi, quindi la CPU basta e avanza.
STEPS_WITH_RESIDENT_MODEL = {"classification"}
PIPELINE_STATE_NAME = "pipeline_state.json"
STATE_SCHEMA_VERSION = 1


def group_steps_by_model(steps: list[str]) -> list[tuple[str | None, list[str]]]:
    """Raggruppa step consecutivi che usano lo stesso modello.

    Esempio: ["language", "analysis", "classification", "review", "tree"]
    -> [("big", ["language", "analysis"]), ("small", ["classification"]), ("big", ["review"]), (None, ["tree"])]

    Usato dalla modalita' --phase-first per minimizzare i cambi di modello:
    invece di avviare il modello 3*N volte (una per ogni fase di ogni documento),
    lo avvia solo una volta per gruppo di fasi consecutive con lo stesso modello.
    """
    groups: list[tuple[str | None, list[str]]] = []
    for step in steps:
        model_key = MODEL_STEPS.get(step)
        if groups and groups[-1][0] == model_key:
            groups[-1][1].append(step)
        else:
            groups.append((model_key, [step]))
    return groups


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def input_signature(input_json: Path) -> dict[str, Any]:
    stat = input_json.stat()
    return {
        "path": str(input_json.resolve()),
        "name": input_json.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def state_path(output_dir: Path) -> Path:
    return output_dir / PIPELINE_STATE_NAME


def load_pipeline_state(output_dir: Path, input_json: Path) -> dict[str, Any]:
    path = state_path(output_dir)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, json.JSONDecodeError):
        print(f"[resume] Stato ignorato perche' non leggibile: {path}")
        return {}

    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        return {}
    if state.get("input") != input_signature(input_json):
        print(f"[resume] Stato ignorato: input cambiato rispetto a {path}")
        return {}
    return state


def write_pipeline_state(output_dir: Path, state: dict[str, Any]) -> None:
    path = state_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def mark_pipeline_step_completed(output_dir: Path, input_json: Path, step: str) -> None:
    state = load_pipeline_state(output_dir, input_json)
    if not state:
        state = {
            "schema_version": STATE_SCHEMA_VERSION,
            "input": input_signature(input_json),
            "output_dir": str(output_dir.resolve()),
            "completed_steps": {},
        }
    state.setdefault("completed_steps", {})[step] = {
        "completed_at": utc_now(),
        "artifacts": [str(path) for path in step_artifact_paths(output_dir, step)],
        # Con quale profilo linguistico e' stato prodotto questo artefatto: se
        # f0 riconosce un'altra lingua, il resume deve rifarlo invece di
        # accettarlo (vedi language_unchanged_since).
        "language_fingerprint": active_language_fingerprint(output_dir),
    }
    write_pipeline_state(output_dir, state)


def active_language_fingerprint(output_dir: Path) -> str | None:
    """Impronta del profilo linguistico scritto da f0 per questo documento."""
    payload = read_json_if_valid(output_dir / DEFAULT_OUTPUT_NAMES["language_config"])
    return payload_digest(payload) if isinstance(payload, dict) else None


def language_unchanged_since(output_dir: Path, recorded_fingerprint: str | None) -> bool:
    """Un artefatto e' ancora valido se il profilo linguistico non e' cambiato.

    ``recorded_fingerprint is None`` significa artefatto prodotto prima
    dell'introduzione di f0, quando le regex italiane erano scritte nel codice:
    equivale alla baseline. Lo stesso vale quando f0 non ha ancora scritto nulla.
    Cosi' un archivio di run italiane non viene invalidato dall'aggiunta dello
    step, ma un documento riconosciuto in un'altra lingua si': i suoi alberi sono
    stati costruiti con le regex sbagliate e vanno rifatti.
    """
    current = active_language_fingerprint(output_dir) or baseline_digest()
    return (recorded_fingerprint or baseline_digest()) == current


def read_json_if_valid(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def artifact_is_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    if path.suffix.lower() == ".json":
        return read_json_if_valid(path) is not None
    return True


def artifacts_exist(output_dir: Path, names: list[str]) -> bool:
    return all(artifact_is_valid(output_dir / name) for name in names)


def artifacts_newer_than_input(output_dir: Path, input_json: Path, names: list[str]) -> bool:
    input_mtime = input_json.stat().st_mtime
    return all((output_dir / name).stat().st_mtime >= input_mtime for name in names)


def step_artifact_names(step: str) -> list[str]:
    names = DEFAULT_OUTPUT_NAMES
    by_step = {
        "language": [names["language_config"]],
        "analysis": [names["analysis"], names["embedding_db"]],
        "classification": [
            names["classification_final"],
            names["classification_slm_raw"],
            names["low_confidence"],
            names["classification_failures"],
        ],
        "review": [names["classification_final"], names["llm_reviewed"]],
        "tree": [
            names["tree"],
            names["rag_graph"],
            names["akn_tree"],
            names["akn_xml"],
            names["tree_validation"],
        ],
    }
    return by_step.get(step, [])


def step_artifact_paths(output_dir: Path, step: str) -> list[Path]:
    return [output_dir / name for name in step_artifact_names(step)]


def count_input_blocks(input_json: Path) -> int:
    data = read_json_if_valid(input_json)
    if data is None:
        return 0
    raw_blocks = data.get("blocks", data if isinstance(data, list) else [])
    count = 0
    for block in raw_blocks:
        if not isinstance(block, dict):
            continue
        text = (block.get("text") or block.get("content") or block.get("testo") or "").strip()
        if text:
            count += 1
    n_file = os.getenv("N_FILE")
    if n_file:
        count = min(count, int(n_file))
    return count


def state_marks_step_completed(output_dir: Path, state: dict[str, Any], step: str) -> bool:
    completed_steps = state.get("completed_steps")
    if not isinstance(completed_steps, dict) or step not in completed_steps:
        return False
    entry = completed_steps[step]
    if step != "language" and not language_unchanged_since(output_dir, entry.get("language_fingerprint")):
        print(f"[resume] Step '{step}' da rifare: il profilo linguistico e' cambiato.")
        return False
    names = step_artifact_names(step)
    return bool(names) and artifacts_exist(output_dir, names)


def legacy_artifacts_mark_step_completed(output_dir: Path, input_json: Path, step: str) -> bool:
    names = step_artifact_names(step)
    if not names or not artifacts_exist(output_dir, names) or not artifacts_newer_than_input(output_dir, input_json, names):
        return False
    if step != "language" and not language_unchanged_since(output_dir, None):
        return False

    if step == "classification":
        raw = read_json_if_valid(output_dir / DEFAULT_OUTPUT_NAMES["classification_slm_raw"])
        failures = read_json_if_valid(output_dir / DEFAULT_OUTPUT_NAMES["classification_failures"])
        if not isinstance(raw, list) or not isinstance(failures, list):
            return False
        return len(raw) + len(failures) >= count_input_blocks(input_json)

    if step == "review" and os.getenv("PROCESS_LOW_CONF_WITH_LLM", "1") == "1":
        final = read_json_if_valid(output_dir / DEFAULT_OUTPUT_NAMES["classification_final"])
        low_confidence = read_json_if_valid(output_dir / DEFAULT_OUTPUT_NAMES["low_confidence"])
        reviewed = read_json_if_valid(output_dir / DEFAULT_OUTPUT_NAMES["llm_reviewed"])
        if not all(isinstance(value, list) for value in (final, low_confidence, reviewed)):
            return False
        return len(reviewed) >= len(low_confidence) and len(final) >= len(read_json_if_valid(output_dir / DEFAULT_OUTPUT_NAMES["classification_slm_raw"]) or [])

    return True


def step_is_completed(output_dir: Path, input_json: Path, step: str, state: dict[str, Any]) -> bool:
    if state_marks_step_completed(output_dir, state, step):
        return True
    return legacy_artifacts_mark_step_completed(output_dir, input_json, step)


def resume_steps(input_json: Path, output_dir: Path, steps: list[str], *, force: bool = False) -> tuple[list[str], list[str]]:
    if force:
        return steps, []

    state = load_pipeline_state(output_dir, input_json)
    skipped = []
    pending = []
    found_missing = False

    for step in steps:
        if not found_missing and step_is_completed(output_dir, input_json, step, state):
            skipped.append(step)
            continue
        pending.append(step)
        # Un f0 da eseguire non invalida di per se' gli step a valle: a deciderlo
        # e' il confronto fra il profilo linguistico che f0 scrive e quello con
        # cui gli artefatti sono stati prodotti (language_unchanged_since).
        # Propagare qui il "manca uno step" rifarebbe l'intera pipeline su ogni
        # run precedente all'introduzione di f0, anche a lingua invariata.
        if step != "language":
            found_missing = True

    return pending, skipped


def run_pre_step_hooks(step: str) -> None:
    if step == "tree":
        stop_model_before_tree()


def apply_embedding_device_policy(step: str, env: dict[str, str]) -> dict[str, str]:
    """Sposta su CPU l'encoder degli step che convivono con un modello acceso.

    Una scelta esplicita dell'utente (``EMBEDDING_DEVICE`` nell'ambiente) vince:
    la politica e' un default sicuro, non un vincolo.
    """
    if step in STEPS_WITH_RESIDENT_MODEL and "EMBEDDING_DEVICE" not in os.environ:
        env["EMBEDDING_DEVICE"] = "cpu"
    return env


def run_step(step: str, env: dict[str, str]) -> None:
    env = apply_embedding_device_policy(step, env)
    module_name = STEP_MODULES[step]
    subprocess.run([sys.executable, "-m", module_name], check=True, env=env)


def execute_step_process(step: str, shared_env: dict[str, str]) -> None:
    model_key = MODEL_STEPS.get(step)
    if model_key:
        with use_model(model_key) as model_env:
            run_step(step, merge_env(shared_env, model_env))
    else:
        run_step(step, shared_env)


def base_env(input_json: Path, output_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    document_id = safe_resource_name(input_json)
    env["BLOCKS_JSON_PATH"] = str(input_json)
    env["OCR_JSON_PATH"] = str(input_json)
    env["OUTPUT_DIR"] = str(output_dir)
    env["DOCUMENT_ID"] = document_id
    env["DOCUMENT_NAME"] = document_key(input_json)
    return env


def merge_env(base: dict[str, str], override: dict[str, str]) -> dict[str, str]:
    merged = override.copy()
    merged.update(base)
    for key in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "LLM_MODEL_NAME", "SLM_MODEL_NAME"):
        if key in override:
            merged[key] = override[key]
    return merged


def expand_step_names(step_names: list[str] | None) -> list[str]:
    if not step_names:
        return DEFAULT_STEPS.copy()

    selected = []
    for raw_name in step_names:
        name = raw_name.lower()
        names = STEP_ALIASES.get(name, [name])
        for step in names:
            if step not in STEP_MODULES:
                valid = sorted(set(STEP_MODULES) | set(STEP_ALIASES))
                raise SystemExit(f"Step non valido: {raw_name}. Valori validi: {', '.join(valid)}")
            if step not in selected:
                selected.append(step)
    return selected


def apply_step_filters(args: argparse.Namespace) -> list[str]:
    steps = expand_step_names(args.step)
    if args.from_step:
        from_steps = expand_step_names([args.from_step])
        first = from_steps[0]
        if first not in DEFAULT_STEPS:
            raise SystemExit(f"--from-step non valido: {args.from_step}")
        # Si riparte da 'first' e si prosegue fino in fondo. Se l'utente ha
        # anche indicato --step, si tengono solo gli step richiesti; senza
        # --step, --from-step da solo significa "da qui in poi, tutto".
        posizione_iniziale = DEFAULT_STEPS.index(first)
        step_da_first_in_poi = DEFAULT_STEPS[posizione_iniziale:]

        steps_filtrati = []
        for step in step_da_first_in_poi:
            if not args.step:
                steps_filtrati.append(step)
            elif step in steps:
                steps_filtrati.append(step)
        steps = steps_filtrati

    skipped = set()
    if args.skip_step:
        skipped = set(expand_step_names(args.skip_step))
    if args.skip_training:
        skipped.update({"classification", "review"})
    return [step for step in steps if step not in skipped]


def execute_steps(
    input_json: Path,
    output_dir: Path,
    steps: list[str],
    shared_env: dict[str, str],
    *,
    on_step_completed=None,
) -> None:
    print(f"[pipeline] Documento: {input_json}")
    print(f"[pipeline] Output: {output_dir}")
    print(f"[pipeline] Step: {', '.join(steps)}")

    for step in steps:
        run_pre_step_hooks(step)
        execute_step_process(step, shared_env)
        if on_step_completed is not None:
            on_step_completed(step)


@dataclass(frozen=True)
class StepStage:
    """Una fase di uno step, eseguita su tutti i documenti prima della successiva."""

    name: str
    env: dict[str, str] = field(default_factory=dict)
    needs_model: bool = True
    stop_model_first: bool = False


def step_stages(step: str, model_key: str | None) -> list[StepStage]:
    """In quante fasi eseguire uno step su tutti i documenti.

    Quasi tutti gli step sono monolitici. ``analysis`` (f1a) no: fa l'analisi
    del campione con l'LLM e poi costruisce il DB di embedding con
    SentenceTransformer. Su memoria unificata i due modelli non coesistono, e
    tenerli nello stesso passo obbligherebbe a spegnere e riaccendere l'LLM una
    volta PER DOCUMENTO — con un avvio di TensorRT-LLM da ~10 minuti, ore.

    Spezzandolo, l'analisi gira per tutti i documenti con il modello acceso, poi
    il modello si spegne UNA volta, poi gli embedding si costruiscono per tutti.
    Su N documenti si passa da N spegnimenti+riaccensioni a uno solo.
    """
    if step != "analysis" or not model_key:
        return [StepStage(name="all", needs_model=model_key is not None)]
    return [
        StepStage(name="analisi LLM", env={"F1A_STAGE": "analysis"}),
        StepStage(
            name="embedding DB",
            env={"F1A_STAGE": "embeddings"},
            needs_model=False,
            stop_model_first=True,
        ),
    ]


def _run_phase_for_all_docs(
    group_steps: list[str],
    inputs: list[Path],
    base_output: Path,
    model_key: str | None,
    model_env: dict[str, str] | None,
    *,
    force: bool = False,
) -> None:
    """Esegue uno step per volta su TUTTI i documenti, poi passa allo step dopo.

    L'ordine e' ``f0 su doc1..docN, poi f1a su doc1..docN, ...``, non
    ``doc1: f0+f1a, doc2: f0+f1a``. E' la ragione d'essere della modalita': con
    l'ordine per documento le fasi si interlacciano e il vantaggio si riduce ai
    soli cambi di modello.

    Il modello e' avviato dal chiamante, ma qui viene RIVERIFICATO prima di ogni
    invocazione. Serve perche' f1a spegne il modello prima di caricare
    l'encoder degli embedding: su una macchina a memoria unificata i due non ci
    stanno insieme, e senza lo spegnimento f1a muore con un CUDA out of memory.
    Riaccenderlo qui costa un riavvio per documento sui soli step che lo
    spengono; gli altri riusano il container gia' attivo senza alcun costo.
    """
    for step in group_steps:
        pending: list[tuple[Path, Path]] = []
        for input_json in inputs:
            out_dir = document_output_dir(base_output, input_json)
            out_dir.mkdir(parents=True, exist_ok=True)
            state = load_pipeline_state(out_dir, input_json)
            if not force and step_is_completed(out_dir, input_json, step, state):
                print(f"[phase-first] Step '{step}' gia' completato per {input_json.name}, salto.")
                continue
            pending.append((input_json, out_dir))

        if not pending:
            continue

        for stage in step_stages(step, model_key):
            if stage.stop_model_first and model_key:
                print(
                    f"[phase-first] Spengo il modello '{model_key}' una volta sola, "
                    f"prima della fase '{stage.name}' di '{step}'.",
                    flush=True,
                )
                stop_model(model_key)

            # Una volta per fase, non per documento: l'hook di 'tree' spegne il
            # modello, e ripeterlo per ogni documento sarebbe solo rumore.
            run_pre_step_hooks(step)

            for input_json, out_dir in pending:
                step_env = base_env(input_json, out_dir)
                if model_env:
                    step_env = merge_env(step_env, model_env)
                step_env.update(stage.env)
                if stage.needs_model and model_key:
                    ensure_model_running(model_key)

                label = f"'{step}'" if stage.name == "all" else f"'{step}' [{stage.name}]"
                print(f"[phase-first] {label} -> {input_json.name}", flush=True)
                run_step(step, step_env)

        for input_json, out_dir in pending:
            mark_pipeline_step_completed(out_dir, input_json, step)


def execute_phase_first(
    inputs: list[Path],
    base_output: Path,
    steps: list[str],
    *,
    force: bool = False,
) -> None:
    """Modalita' phase-first: minimizza i cambi di modello raggruppando i documenti per fase.

    Modalita' default (per documento)::

        doc1: f1a(big) -> f1b(small) -> f1c(big)   <- 3 avvii modello
        doc2: f1a(big) -> f1b(small) -> f1c(big)   <- 3 avvii modello
        Totale per N doc: 3*N avvii

    Modalita' phase-first::

        f0(big)    -> doc1, doc2, ...
        f1a(big)   -> doc1, doc2, ...   <- stesso modello del gruppo: nessun cambio
        f1b(small) -> doc1, doc2, ...   <- 1 cambio
        f1c(big)   -> doc1, doc2, ...   <- 1 cambio
        f2         -> doc1, doc2, ...   <- nessun modello
        Totale: 2 cambi di modello (indipendente da N)

    Uno step viene completato su TUTTI i documenti prima di passare al
    successivo. Gli step con lo stesso modello stanno in un unico gruppo e ne
    condividono l'istanza; ``ensure_model_running`` la riverifica prima di ogni
    invocazione, cosi' uno step che spegne il modello per liberare memoria (f1a,
    prima di caricare l'encoder degli embedding) non lascia a secco i documenti
    successivi.

    Il resume (pipeline_state.json) e' pienamente supportato.
    """
    groups = group_steps_by_model(steps)
    n_docs = len(inputs)
    n_groups = len(groups)
    print(f"[phase-first] {n_groups} gruppo/i di fasi x {n_docs} documento/i.", flush=True)
    for group_idx, (model_key, group_steps) in enumerate(groups, 1):
        per_doc_steps = group_steps
        label = f"[{', '.join(group_steps)}] (modello: {model_key or 'nessuno'})"
        print(f"[phase-first] Avvio gruppo {group_idx}/{n_groups}: {label}", flush=True)

        if per_doc_steps:
            if model_key:
                with use_model(model_key) as model_env:
                    _run_phase_for_all_docs(per_doc_steps, inputs, base_output, model_key, model_env, force=force)
            else:
                _run_phase_for_all_docs(per_doc_steps, inputs, base_output, None, None, force=force)

        print(f"[phase-first] Gruppo {group_idx}/{n_groups} completato: {label}", flush=True)



def run_for_document(
    input_json: Path,
    output_dir: Path,
    steps: list[str],
    force: bool = False,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    executed: list[str] = []

    # f0 va eseguito e valutato PRIMA di decidere il resume del resto: e' lui a
    # scrivere il profilo linguistico, e il profilo decide se gli artefatti gia'
    # presenti sono ancora validi. Calcolare tutto in un colpo solo userebbe il
    # profilo vecchio (o nessuno) per giudicare gli step a valle.
    if "language" in steps:
        language_pending, language_skipped = resume_steps(input_json, output_dir, ["language"], force=force)
        if language_skipped:
            print(f"[resume] Step gia' completati per {input_json.name}: language")
        if language_pending:
            language_env = base_env(input_json, output_dir)
            execute_steps(input_json, output_dir, ["language"], language_env)
            mark_pipeline_step_completed(output_dir, input_json, "language")
            executed.append("language")
        steps = [step for step in steps if step != "language"]

    pending_steps, skipped_steps = resume_steps(input_json, output_dir, steps, force=force)
    if skipped_steps:
        print(f"[resume] Step gia' completati per {input_json.name}: {', '.join(skipped_steps)}")
    if not pending_steps:
        print(f"[resume] Documento gia' completo per gli step richiesti, salto: {input_json}")
        return executed

    shared_env = base_env(input_json, output_dir)

    def segna_step_completato(step: str) -> None:
        mark_pipeline_step_completed(output_dir, input_json, step)

    execute_steps(
        input_json,
        output_dir,
        pending_steps,
        shared_env,
        on_step_completed=segna_step_completato,
    )
    return executed + pending_steps


def iter_inputs(args: argparse.Namespace) -> list[Path]:
    if args.input:
        input_path = Path(args.input)
        if input_path.exists():
            return [input_path.resolve()]

        hints = []
        raw_input = args.input
        if "\\" in raw_input:
            bash_path = raw_input.replace("\\", "/")
            if bash_path.startswith("./"):
                hints.append(f"Su Linux non usare backslash: prova --input {bash_path}")
            elif bash_path.startswith("."):
                hints.append(f"Su Linux non usare backslash: prova --input ./{bash_path.lstrip('.')}")
            else:
                hints.append(f"Su Linux non usare backslash: prova --input {bash_path}")
        if "\\" not in raw_input and raw_input.startswith(".") and not raw_input.startswith("./"):
            visible_candidate = Path("./" + raw_input.lstrip("."))
            hints.append(f"Hai passato '{raw_input}'. Su Linux usa './{raw_input.lstrip('.')}' se il file e' nella cartella corrente.")
            if visible_candidate.exists():
                hints.append(f"File trovato come: {visible_candidate}")
        parent_candidate = Path("..") / input_path.name.lstrip(".")
        if parent_candidate.exists():
            hints.append(f"File trovato nella cartella superiore: --input {parent_candidate}")

        message = f"File input non trovato: {input_path}"
        if hints:
            message += "\n" + "\n".join(hints)
        raise SystemExit(message)

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Cartella input non trovata: {input_dir}")
    return [path.resolve() for path in sorted(input_dir.glob(args.pattern))]


def main() -> None:
    parser = argparse.ArgumentParser(description="LegisLeaF: orchestratore delle fasi F0 -> F1 -> F2.")
    parser.add_argument("--input", help="Singolo JSON OCR/major voting da processare.")
    parser.add_argument("--input-dir", default="input_docs", help="Cartella con piu' JSON da processare.")
    parser.add_argument("--pattern", default="*.json", help="Pattern dei file in --input-dir.")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Cartella base per gli output (relativa alla CWD).",
    )
    parser.add_argument(
        "--step",
        action="append",
        help="Step da eseguire. Valori: all, preprocessing, language, analysis, classification, review, tree, f0, f1, f1a, f1b, f1c, f2.",
    )
    parser.add_argument("--from-step", help="Esegue dal primo step indicato in poi.")
    parser.add_argument("--skip-step", action="append", help="Step da saltare; accetta gli stessi valori di --step.")
    parser.add_argument("--skip-training", action="store_true", help="Compatibilita': salta classificazione e revisione modello.")
    parser.add_argument("--force", action="store_true", help="Riesegue gli step richiesti anche se gli output risultano gia' presenti.")
    parser.add_argument(
        "--phase-first",
        action="store_true",
        help=(
            "Modalita' phase-first: esegui ogni fase su tutti i documenti prima di cambiare modello. "
            "Riduce i cambi di modello da 3*N a 2 per le fasi f1a/f1b/f1c."
        ),
    )
    args = parser.parse_args()

    steps = apply_step_filters(args)
    if not steps:
        raise SystemExit("Nessuno step da eseguire dopo l'applicazione dei filtri.")

    inputs = iter_inputs(args)
    if not inputs:
        raise SystemExit("Nessun documento trovato.")

    base_output = Path(args.output_dir).resolve()

    if args.phase_first:
        model_steps_in_run = [s for s in steps if MODEL_STEPS.get(s)]
        if model_steps_in_run:
            groups_preview = group_steps_by_model(steps)
            model_switches = sum(1 for i in range(1, len(groups_preview)) if groups_preview[i][0] != groups_preview[i - 1][0])
            print(
                f"[phase-first] Modalita' attiva: {len(model_steps_in_run)} step con modello, "
                f"{model_switches} cambio/i di modello totali (invece di {len(model_steps_in_run) * len(inputs)})."
            )
        execute_phase_first(inputs, base_output, steps, force=args.force)
    else:
        for input_json in inputs:
            out_dir = document_output_dir(base_output, input_json)
            run_for_document(input_json, out_dir, steps, force=args.force)


if __name__ == "__main__":
    main()
