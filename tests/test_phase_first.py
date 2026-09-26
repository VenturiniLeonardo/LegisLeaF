"""Test dell'ordine di esecuzione in modalita' --phase-first.

L'ordine era regredito in silenzio: il ciclo esterno era sui documenti invece
che sugli step, quindi la modalita' faceva ``doc1: f0+f1a, doc2: f0+f1a``
mentre il suo stesso docstring prometteva ``f0 su tutti, poi f1a su tutti``.
Nessun test lo copriva, e la differenza non produce errori — solo un ordine
diverso da quello dichiarato.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from legisleaf import run as run_pipeline


@pytest.fixture()
def documenti(tmp_path: Path) -> list[Path]:
    percorsi = []
    for nome in ("doc_a", "doc_b", "doc_c"):
        path = tmp_path / f"{nome}.json"
        path.write_text("[]", encoding="utf-8")
        percorsi.append(path)
    return percorsi


@pytest.fixture()
def tracciato(monkeypatch) -> list[tuple[str, str, str]]:
    """Registra ogni invocazione come ``(step, documento, fase)`` senza eseguire nulla."""
    eventi: list[tuple[str, str, str]] = []

    def fake_run_step(step: str, env: dict) -> None:
        eventi.append((step, Path(env["BLOCKS_JSON_PATH"]).stem, env.get("F1A_STAGE", "-")))

    monkeypatch.setattr(run_pipeline, "run_step", fake_run_step)
    monkeypatch.setattr(run_pipeline, "mark_pipeline_step_completed", lambda *a, **k: None)
    monkeypatch.setattr(run_pipeline, "step_is_completed", lambda *a, **k: False)
    monkeypatch.setattr(run_pipeline, "run_pre_step_hooks", lambda step: None)
    monkeypatch.setattr(run_pipeline, "stop_model", lambda *a, **k: None)
    return eventi


def test_uno_step_per_volta_su_tutti_i_documenti(documenti, tracciato, tmp_path, monkeypatch):
    monkeypatch.setattr(run_pipeline, "ensure_model_running", lambda key: {})

    run_pipeline._run_phase_for_all_docs(
        ["language", "classification"], documenti, tmp_path / "out", "big", {"OPENAI_BASE_URL": "x"}
    )

    assert tracciato == [
        ("language", "doc_a", "-"),
        ("language", "doc_b", "-"),
        ("language", "doc_c", "-"),
        ("classification", "doc_a", "-"),
        ("classification", "doc_b", "-"),
        ("classification", "doc_c", "-"),
    ]


# --------------------------------------------------------------------------- #
# Spegnimento del modello: uno solo per l'intero batch                         #
# --------------------------------------------------------------------------- #
def test_f1a_analizza_tutti_poi_spegne_una_volta_poi_calcola_gli_embedding(
    documenti, tracciato, tmp_path, monkeypatch
):
    """Un avvio di TensorRT-LLM costa ~10 minuti: va speso una volta, non N."""
    spegnimenti: list[str] = []
    monkeypatch.setattr(run_pipeline, "ensure_model_running", lambda key: {})
    monkeypatch.setattr(run_pipeline, "stop_model", lambda key, *a, **k: spegnimenti.append(key))

    run_pipeline._run_phase_for_all_docs(["analysis"], documenti, tmp_path / "out", "big", {})

    assert tracciato == [
        ("analysis", "doc_a", "analysis"),
        ("analysis", "doc_b", "analysis"),
        ("analysis", "doc_c", "analysis"),
        ("analysis", "doc_a", "embeddings"),
        ("analysis", "doc_b", "embeddings"),
        ("analysis", "doc_c", "embeddings"),
    ]
    assert spegnimenti == ["big"]


def test_la_fase_embedding_non_riaccende_il_modello(documenti, tracciato, tmp_path, monkeypatch):
    """Riaccenderlo per gli embedding annullerebbe il risparmio."""
    accensioni: list[str] = []
    monkeypatch.setattr(run_pipeline, "ensure_model_running", lambda key: accensioni.append(key) or {})

    run_pipeline._run_phase_for_all_docs(["analysis"], documenti, tmp_path / "out", "big", {})

    # Una verifica per documento nella fase LLM, nessuna nella fase embedding.
    assert accensioni == ["big", "big", "big"]


def test_senza_modello_f1a_resta_monolitico(tmp_path):
    """In modalita' per-documento non c'e' niente da spezzare."""
    assert [s.name for s in run_pipeline.step_stages("analysis", None)] == ["all"]
    assert [s.name for s in run_pipeline.step_stages("analysis", "big")] == ["analisi LLM", "embedding DB"]
    assert [s.name for s in run_pipeline.step_stages("review", "big")] == ["all"]


def test_senza_modello_non_si_tocca_docker(documenti, tracciato, tmp_path, monkeypatch):
    def esplodi(key):
        raise AssertionError("ensure_model_running non va chiamato per gli step senza modello")

    monkeypatch.setattr(run_pipeline, "ensure_model_running", esplodi)

    run_pipeline._run_phase_for_all_docs(["tree"], documenti, tmp_path / "out", None, None)

    assert tracciato == [
        ("tree", "doc_a", "-"),
        ("tree", "doc_b", "-"),
        ("tree", "doc_c", "-"),
    ]


def test_i_documenti_gia_completati_sono_saltati(documenti, tracciato, tmp_path, monkeypatch):
    monkeypatch.setattr(run_pipeline, "ensure_model_running", lambda key: {})
    monkeypatch.setattr(
        run_pipeline,
        "step_is_completed",
        lambda out_dir, input_json, step, state: input_json.stem == "doc_b",
    )

    run_pipeline._run_phase_for_all_docs(["language"], documenti, tmp_path / "out", "big", {})

    assert tracciato == [("language", "doc_a", "-"), ("language", "doc_c", "-")]


def test_i_gruppi_seguono_il_modello_non_lo_step():
    """Step consecutivi con lo stesso modello condividono l'istanza."""
    groups = run_pipeline.group_steps_by_model(run_pipeline.DEFAULT_STEPS)
    assert groups == [
        ("big", ["language", "analysis"]),
        ("small", ["classification"]),
        ("big", ["review"]),
        (None, ["tree"]),
    ]
