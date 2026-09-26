"""Test del resume di run_pipeline dopo l'introduzione dello step f0.

Aggiungere uno step all'inizio della pipeline rischia di invalidare tutti gli
artefatti gia' prodotti: ``resume_steps`` considera pendente ogni step che segue
il primo mancante. Questi test fissano il comportamento voluto:

* una run precedente a f0 resta valida, va eseguito solo f0;
* se f0 conferma la lingua, non si rifa' nulla;
* se f0 riconosce un'altra lingua, tutto il resto va rifatto — quegli artefatti
  sono stati costruiti con le regex sbagliate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from legisleaf import run as rp
from legisleaf.language import bundled_config_path

ALL_STEPS = rp.DEFAULT_STEPS
DOWNSTREAM = [step for step in ALL_STEPS if step != "language"]


@pytest.fixture()
def run_dir(tmp_path: Path) -> tuple[Path, Path]:
    """Cartella di output con tutti gli artefatti degli step, tranne quello di f0."""
    input_json = tmp_path / "documento.json"
    input_json.write_text(json.dumps({"blocks": [{"text": "Art. 1"}]}), encoding="utf-8")

    out = tmp_path / "documento_output"
    out.mkdir()
    # I controlli di completezza "legacy" contano gli elementi di alcuni file
    # (blocchi classificati vs blocchi in ingresso, revisioni vs bassa
    # confidenza): servono liste coerenti, non JSON vuoti qualsiasi.
    names = rp.DEFAULT_OUTPUT_NAMES
    contents = {
        names["classification_slm_raw"]: [{}],
        names["classification_final"]: [{}],
    }
    for step in DOWNSTREAM:
        for name in rp.step_artifact_names(step):
            path = out / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(contents.get(name, [])), encoding="utf-8")
    return input_json, out


def mark_all_downstream_completed(out: Path, input_json: Path) -> None:
    for step in DOWNSTREAM:
        rp.mark_pipeline_step_completed(out, input_json, step)


def write_language_config(out: Path, *, code: str = "it") -> None:
    payload = json.loads(bundled_config_path("it").read_text(encoding="utf-8"))
    payload["code"] = code
    (out / "language_config.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def strip_fingerprints(out: Path) -> None:
    """Riporta lo stato al formato scritto prima dell'introduzione di f0."""
    path = rp.state_path(out)
    state = json.loads(path.read_text(encoding="utf-8"))
    for entry in state["completed_steps"].values():
        entry.pop("language_fingerprint", None)
    path.write_text(json.dumps(state), encoding="utf-8")


def test_run_precedente_a_f0_esegue_solo_f0(run_dir):
    input_json, out = run_dir
    mark_all_downstream_completed(out, input_json)
    strip_fingerprints(out)

    pending, skipped = rp.resume_steps(input_json, out, ALL_STEPS)
    assert pending == ["language"]
    assert skipped == DOWNSTREAM


def test_lingua_confermata_non_rifa_nulla(run_dir):
    input_json, out = run_dir
    write_language_config(out, code="it")
    mark_all_downstream_completed(out, input_json)
    rp.mark_pipeline_step_completed(out, input_json, "language")

    pending, _ = rp.resume_steps(input_json, out, ALL_STEPS)
    assert pending == []


def test_cambio_di_lingua_invalida_tutti_gli_step_a_valle(run_dir):
    input_json, out = run_dir
    write_language_config(out, code="it")
    mark_all_downstream_completed(out, input_json)
    rp.mark_pipeline_step_completed(out, input_json, "language")

    write_language_config(out, code="en")
    rp.mark_pipeline_step_completed(out, input_json, "language")

    pending, skipped = rp.resume_steps(input_json, out, ALL_STEPS)
    assert skipped == ["language"]
    assert pending == DOWNSTREAM


def test_artefatti_legacy_senza_stato_richiedono_la_baseline(run_dir):
    """Senza pipeline_state.json vale la stessa regola, dedotta dai soli artefatti."""
    input_json, out = run_dir
    assert not rp.state_path(out).exists()

    pending, _ = rp.resume_steps(input_json, out, ALL_STEPS)
    assert pending == ["language"], "artefatti italiani preesistenti: rifare solo f0"

    # Con il profilo gia' scritto f0 risulta fatto, ma il resto no: quegli
    # artefatti sono stati costruiti con le regex italiane.
    write_language_config(out, code="en")
    pending, skipped = rp.resume_steps(input_json, out, ALL_STEPS)
    assert skipped == ["language"]
    assert pending == DOWNSTREAM, "profilo non piu' italiano: gli artefatti non sono riusabili"
