"""Test del ciclo di vita dei container dei modelli.

Il caso che ha motivato questi test: finche' i due modelli condividevano un solo
container, avviarne uno rimuoveva l'altro perche' era letteralmente lo stesso
container, e la mutua esclusione era un effetto collaterale della configurazione.
Passando a container distinti quella garanzia sparisce in silenzio — nessun
errore, solo due server di modello residenti insieme e un out of memory al
secondo avvio su hardware a memoria unificata.
"""

from __future__ import annotations

import pytest

from legisleaf import model_manager as dmm
from legisleaf.settings import MODEL_RUNTIMES


def test_i_due_runtime_hanno_container_distinti():
    nomi = {r.key: r.container_name for r in MODEL_RUNTIMES.values()}
    assert len(set(nomi.values())) == len(nomi), f"container condiviso fra runtime: {nomi}"


def test_entrambi_i_runtime_sono_esclusivi():
    """Su memoria unificata due server di modello non coesistono."""
    for runtime in MODEL_RUNTIMES.values():
        assert runtime.exclusive, f"runtime {runtime.key} non esclusivo"


def test_avviare_un_runtime_ferma_l_altro(monkeypatch):
    fermati: list[str] = []
    monkeypatch.setattr(dmm, "inspect_container_running", lambda nome: True)
    monkeypatch.setattr(dmm, "stop_container", lambda nome: fermati.append(nome))

    dmm.stop_conflicting_runtimes("small")
    assert fermati == [MODEL_RUNTIMES["big"].container_name]

    fermati.clear()
    dmm.stop_conflicting_runtimes("big")
    assert fermati == [MODEL_RUNTIMES["small"].container_name]


def test_non_ferma_un_container_gia_spento(monkeypatch):
    """Un ``docker stop`` inutile e' rumore nel log, non un errore, ma va evitato."""
    fermati: list[str] = []
    monkeypatch.setattr(dmm, "inspect_container_running", lambda nome: False)
    monkeypatch.setattr(dmm, "stop_container", lambda nome: fermati.append(nome))
    assert dmm.stop_conflicting_runtimes("small") == []
    assert fermati == []


def test_container_inesistente_non_e_un_errore(monkeypatch):
    """``inspect`` ritorna None quando il container non esiste ancora."""
    monkeypatch.setattr(dmm, "inspect_container_running", lambda nome: None)
    monkeypatch.setattr(dmm, "stop_container", lambda nome: pytest.fail("non deve fermare nulla"))
    assert dmm.stop_conflicting_runtimes("big") == []


def test_un_runtime_non_esclusivo_non_ferma_nessuno(monkeypatch):
    """Se un giorno i modelli stessero in memoria insieme, il flag lo dichiara."""
    import dataclasses

    non_esclusivo = dataclasses.replace(MODEL_RUNTIMES["small"], exclusive=False)
    monkeypatch.setitem(dmm.MODEL_RUNTIMES, "small", non_esclusivo)
    monkeypatch.setattr(dmm, "inspect_container_running", lambda nome: True)
    monkeypatch.setattr(dmm, "stop_container", lambda nome: pytest.fail("non deve fermare nulla"))
    assert dmm.stop_conflicting_runtimes("small") == []


def test_il_container_non_viene_rimosso_ma_solo_fermato():
    """Rimuoverlo costringerebbe a un compose up e a ricaricare il modello da zero."""
    for runtime in MODEL_RUNTIMES.values():
        assert not runtime.remove_before_start, f"{runtime.key}: rimozione prima dell'avvio"
        assert not runtime.cleanup_after, f"{runtime.key}: rimozione dopo l'uso"
