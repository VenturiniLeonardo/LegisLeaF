import os
import re
from dataclasses import dataclass
from pathlib import Path


# Endpoint OpenAI-compatibile (vLLM, TensorRT-LLM, llama.cpp server, OpenRouter...).
# Sovrascrivibili con le variabili omonime (vedi .env.example).
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8080/v1")
# Un server locale accetta una chiave finta; per un servizio remoto va messa la
# chiave vera nella variabile d'ambiente, mai qui.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY") or "dummy"

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME_DEFAULT", "nvidia/nemotron-3-super")
# Nome con cui il server espone lo SLM su /v1/models (il --served_model_name),
# non necessariamente l'id Hugging Face. Il default corrisponde a
# Qwen3-30B-A3B-Instruct-2507 servito come 'qwen3-classifier'.
SLM_MODEL_NAME = os.getenv("SLM_MODEL_NAME_DEFAULT", "qwen3-classifier")

BIG_MODEL_COMPOSE_SERVICE = os.getenv("BIG_MODEL_COMPOSE_SERVICE", "tensorrt-llm")
SMALL_MODEL_COMPOSE_SERVICE = os.getenv("SMALL_MODEL_COMPOSE_SERVICE", "tensorrt-llm-qwen3")

# I due modelli hanno container DISTINTI. Prima ne condividevano uno solo, e
# quella coincidenza era l'unica cosa che impediva loro di coesistere: avviare il
# piccolo rimuoveva il grande perche' era letteralmente lo stesso container. Con
# container separati quella mutua esclusione implicita non c'e' piu', e su una
# macchina a memoria unificata nemotron piu' un 80B non ci stanno insieme: la
# garanzia diventa esplicita (vedi ModelRuntime.exclusive).
TENSORRT_CONTAINER_NAME = os.getenv("BIG_MODEL_CONTAINER_NAME", "tensorrt-llm-nemotron")
SMALL_MODEL_CONTAINER_NAME = os.getenv("SMALL_MODEL_CONTAINER_NAME", "tensorrt-llm-qwen3")
MODEL_READY_TIMEOUT_SECONDS = 900
# Gestione opzionale dei server di modello via Docker Compose: attiva solo se
# DOCKER_COMPOSE_FILE punta a un compose che definisce i servizi sopra.
DOCKER_COMPOSE_FILE = os.getenv("DOCKER_COMPOSE_FILE", "")


@dataclass(frozen=True)
class ModelRuntime:
    key: str
    compose_service: str | None
    container_name: str
    model_env_var: str
    model_name: str
    managed_by_compose: bool = True
    cleanup_after: bool = True
    remove_before_start: bool = True
    # Il runtime non puo' condividere la memoria con gli altri: prima di avviarlo
    # si fermano i loro container. Su hardware a memoria unificata e' la norma,
    # non l'eccezione.
    exclusive: bool = True


# Con containers gestiti da un altro utente/contesto (docker.sock non
# raggiungibile da questo account) ensure_model_running
# e use_model proverebbero comunque 'docker inspect'/'docker compose up' e
# fallirebbero, anche quando il modello giusto e' gia' in ascolto: 'docker
# inspect' fallito viene letto come "container fermo", non come "non posso
# vedere lo stato", e si finisce a un 'docker compose up' che non funziona per
# noi. EXTERNAL_MODEL_MANAGEMENT=1 disattiva managed_by_compose per entrambi i
# runtime: si passa dritti al solo controllo HTTP su /v1/models, lo swap va
# fatto a mano da chi ha accesso a Docker. Senza DOCKER_COMPOSE_FILE il default
# e' la gestione esterna: il framework si limita ad attendere l'endpoint.
_EXTERNAL_MODEL_MANAGEMENT = os.getenv("EXTERNAL_MODEL_MANAGEMENT", "0" if DOCKER_COMPOSE_FILE else "1") == "1"

MODEL_RUNTIMES = {
    "big": ModelRuntime(
        key="big",
        compose_service=BIG_MODEL_COMPOSE_SERVICE,
        container_name=TENSORRT_CONTAINER_NAME,
        model_env_var="LLM_MODEL_NAME",
        model_name=LLM_MODEL_NAME,
        managed_by_compose=not _EXTERNAL_MODEL_MANAGEMENT,
        cleanup_after=False,
        remove_before_start=False,
    ),
    "small": ModelRuntime(
        key="small",
        compose_service=SMALL_MODEL_COMPOSE_SERVICE,
        container_name=SMALL_MODEL_CONTAINER_NAME,
        model_env_var="SLM_MODEL_NAME",
        model_name=SLM_MODEL_NAME,
        managed_by_compose=not _EXTERNAL_MODEL_MANAGEMENT,
        # Con un container proprio non serve piu' ricrearlo a ogni avvio: un
        # 'docker start' su un container esistente evita il caricamento del
        # modello da zero, che su un 80B e' la parte costosa.
        cleanup_after=False,
        remove_before_start=False,
    ),
}


DEFAULT_OUTPUT_NAMES = {
    "language_config": "language_config.json",
    "analysis": "output_analysis.json",
    "embedding_db": "embedding_examples_db.npz",
    "classification_final": "classificazione_blocchi.json",
    "classification_slm_raw": "classificazione_blocchi_slm_raw.json",
    "low_confidence": "classificazione_bassa_confidenza.json",
    "llm_reviewed": "classificazione_bassa_confidenza_llm.json",
    "classification_failures": "classificazione_fallimenti.json",
    "llm_review_raw_errors": "llm_review_raw_errors.json",
    "tree": "struttura_ad_albero.json",
    "rag_graph": "struttura_rag_graph.json",
    "akn_tree": "struttura_akn.json",
    "akn_xml": "documento_akn.xml",
    "tree_validation": "tree_validation_report.json",
}


def document_output_dir(base_output_dir: str | Path, input_json: str | Path) -> Path:
    return Path(base_output_dir) / f"{document_key(input_json)}_output"


def document_key(input_json: str | Path) -> str:
    return Path(input_json).stem


def safe_resource_name(input_json: str | Path, *, prefix: str = "") -> str:
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", document_key(input_json)).strip("_")
    name = re.sub(r"_+", "_", name)
    if not name:
        name = "document"
    if not re.match(r"^[A-Za-z_]", name):
        name = f"doc_{name}"
    return f"{prefix}{name}"
