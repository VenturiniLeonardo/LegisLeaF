import os
import json
import http.client
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from legisleaf.settings import (
        DOCKER_COMPOSE_FILE,
        MODEL_READY_TIMEOUT_SECONDS,
        MODEL_RUNTIMES,
        OPENAI_API_KEY,
        OPENAI_BASE_URL,
    )


def run_command(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=check)


def inspect_container_running(container_name: str) -> bool | None:
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip().lower() == "true"


def remove_container(container_name: str) -> None:
    run_command(["docker", "stop", container_name], check=False)
    run_command(["docker", "rm", container_name], check=False)


def stop_container(container_name: str) -> None:
    if not any(r.managed_by_compose for r in MODEL_RUNTIMES.values()):
        return  # runtime esterni: non tocchiamo Docker (potrebbe non esserci)
    print(f"[model-manager] docker stop {container_name}", flush=True)
    run_command(["docker", "stop", container_name], check=False)


def stop_big_model(reason: str = "libero risorse") -> None:
    runtime = MODEL_RUNTIMES["big"]
    print(f"[model-manager] Stop runtime 'big': {reason}.", flush=True)
    stop_container(runtime.container_name)


def start_compose_service(service_name: str) -> None:
    compose_file = os.getenv("DOCKER_COMPOSE_FILE", DOCKER_COMPOSE_FILE)
    print(f"[model-manager] docker compose -f {compose_file} up -d {service_name}", flush=True)
    run_command(["docker", "compose", "-f", compose_file, "up", "-d", service_name], check=True)


def start_or_reuse_container(container_name: str) -> bool:
    running = inspect_container_running(container_name)
    if running is None:
        return False
    if running:
        print(f"[model-manager] Container gia' attivo: {container_name}.", flush=True)
        return True
    print(f"[model-manager] Container esistente fermo: docker start {container_name}", flush=True)
    run_command(["docker", "start", container_name], check=True)
    return True


def stop_conflicting_runtimes(model_key: str) -> list[str]:
    """Ferma i container degli altri runtime prima di avviare questo.

    Serve da quando i due modelli hanno container distinti. Prima ne condividevano
    uno: avviare il piccolo rimuoveva il grande perche' era lo stesso container, e
    la mutua esclusione era un effetto collaterale della configurazione. Con
    container separati due server di modello resterebbero residenti insieme, e su
    memoria unificata il secondo avvio muore con un out of memory.

    Si ferma senza rimuovere: il riavvio successivo e' un ``docker start`` e non un
    ``compose up``, che su un 80B fa la differenza fra secondi e minuti.
    """
    runtime = MODEL_RUNTIMES[model_key]
    if not getattr(runtime, "exclusive", True):
        return []

    fermati = []
    for altro in MODEL_RUNTIMES.values():
        if altro.key == model_key or altro.container_name == runtime.container_name:
            continue
        if inspect_container_running(altro.container_name):
            print(
                f"[model-manager] Runtime '{altro.key}' attivo su un container diverso "
                f"({altro.container_name}): lo fermo per liberare memoria.",
                flush=True,
            )
            stop_container(altro.container_name)
            fermati.append(altro.container_name)
    return fermati


def stop_model(model_key: str, reason: str = "libero risorse") -> None:
    """Ferma il container di un runtime, senza rimuoverlo.

    Il container fermo viene riavviato da ``ensure_model_running`` con
    ``docker start``, che e' molto piu' rapido di un ``compose up`` da zero.
    """
    runtime = MODEL_RUNTIMES[model_key]
    print(f"[model-manager] Stop runtime '{model_key}': {reason}.", flush=True)
    stop_container(runtime.container_name)


def stop_model_before_tree() -> None:
    stop_big_model("prima dello step tree")


def wait_openai_server(
    timeout_seconds: int = MODEL_READY_TIMEOUT_SECONDS,
    interval_seconds: int = 5,
    expected_model: str | None = None,
) -> None:
    base_url = os.getenv("OPENAI_BASE_URL", OPENAI_BASE_URL)
    models_url = f"{base_url.rstrip('/')}/models"
    deadline = time.time() + timeout_seconds
    next_log_at = 0.0
    last_status = "endpoint non ancora raggiunto"
    require_model_match = os.getenv("REQUIRE_MODEL_ID_MATCH", "0") == "1"

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(models_url, timeout=10) as response:
                if response.status == 200:
                    if expected_model and require_model_match:
                        payload = json.loads(response.read().decode("utf-8"))
                        model_ids = {item.get("id") for item in payload.get("data", []) if isinstance(item, dict)}
                        if model_ids and expected_model not in model_ids:
                            last_status = f"endpoint ok, modello atteso non presente: {expected_model}"
                            time.sleep(interval_seconds)
                            continue
                    print(f"[model-manager] Endpoint pronto: {models_url}", flush=True)
                    return
        except (
            json.JSONDecodeError,
            urllib.error.URLError,
            TimeoutError,
            http.client.HTTPException,
            OSError,
        ) as exc:
            last_status = str(exc) or exc.__class__.__name__

        now = time.time()
        if now >= next_log_at:
            remaining = max(0, int(deadline - now))
            print(f"[model-manager] Attesa endpoint modello: {models_url} | rimanenti {remaining}s | {last_status}", flush=True)
            next_log_at = now + 30

        time.sleep(interval_seconds)

    raise TimeoutError(f"Model server not ready at {models_url}")


def model_env(model_key: str) -> dict[str, str]:
    runtime = MODEL_RUNTIMES[model_key]
    env = os.environ.copy()
    env["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", OPENAI_BASE_URL)
    env["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
    env[runtime.model_env_var] = runtime.model_name
    return env


def ensure_model_running(model_key: str) -> dict[str, str]:
    """Garantisce che il modello sia in ascolto, riavviandolo se qualcuno l'ha spento.

    Diverso da ``use_model``: non rimuove mai il container prima di partire
    (quello e' un reset, non una verifica) e non lo ferma all'uscita. Serve nei
    cicli che eseguono uno step su molti documenti, dove uno step puo' aver
    spento il modello per liberare memoria — f1a lo fa prima di caricare
    l'encoder degli embedding. Se il container e' gia' attivo il costo e' la
    sola chiamata a ``/v1/models``.
    """
    runtime = MODEL_RUNTIMES[model_key]
    if runtime.managed_by_compose:
        stop_conflicting_runtimes(model_key)
        if not start_or_reuse_container(runtime.container_name):
            if runtime.compose_service is None:
                raise ValueError(f"Runtime {model_key} non ha compose_service configurato.")
            start_compose_service(runtime.compose_service)
    wait_openai_server(expected_model=runtime.model_name)
    return model_env(model_key)


@contextmanager
def use_model(model_key: str, *, cleanup_after: bool | None = None):
    runtime = MODEL_RUNTIMES[model_key]

    if runtime.managed_by_compose:
        print(f"[model-manager] Runtime '{model_key}' gestito via compose: avvio {runtime.compose_service}.", flush=True)
        stop_conflicting_runtimes(model_key)
        if runtime.remove_before_start:
            remove_container(runtime.container_name)
        if runtime.compose_service is None:
            raise ValueError(f"Runtime {model_key} non ha compose_service configurato.")
        if runtime.remove_before_start or not start_or_reuse_container(runtime.container_name):
            start_compose_service(runtime.compose_service)
    else:
        print(f"[model-manager] Runtime '{model_key}' esterno: attendo endpoint gia' attivo.", flush=True)

    wait_openai_server(expected_model=runtime.model_name)

    env = os.environ.copy()
    env["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", OPENAI_BASE_URL)
    env["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
    env[runtime.model_env_var] = runtime.model_name

    try:
        yield env
    finally:
        effective_cleanup = runtime.cleanup_after if cleanup_after is None else cleanup_after
        if runtime.managed_by_compose and effective_cleanup:
            remove_container(runtime.container_name)
