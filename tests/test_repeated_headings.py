"""Test della soppressione di indice, testatine e intestazioni duplicate (f2).

Il caso che ha motivato il codice: nel CFR "§ 164.512" generava 8 nodi per 41
numeri distinti contro 39 del golden, e in data_act "Schedule 2" ne generava 14.
Erano voci d'indice, testatine di pagina e rimandi, tutti indistinguibili da
un'intestazione vera guardando il solo testo.

I test fissano anche cio' che NON deve essere soppresso: un allegato puo'
legittimamente contenere una propria "Parte 1", e una partizione realmente
priva di testo non deve sparire solo perche' il suo numero si ripete.
"""

from __future__ import annotations

from legisleaf.f2.build_tree import detect_repeated_headings, record_key


def blocco(page: int, page_order: int, testo: str, ordine: int | None = None) -> dict:
    return {
        "file": "doc.json",
        "page": page,
        "page_order": page_order,
        "ordine": ordine if ordine is not None else page * 100 + page_order,
        "testo": testo,
        "etichetta": "content",
    }


def soppressi(records: list[dict]) -> set:
    return detect_repeated_headings(records)[0]


def declassati(records: list[dict]) -> set:
    return detect_repeated_headings(records)[1]


def test_pagina_di_indice_soppressa():
    """Una pagina fatta quasi solo di intestazioni e' un indice, non struttura."""
    indice = [blocco(1, i, f"Art. {i + 1} Titolo") for i in range(6)]
    corpo = [blocco(2, 0, "Art. 1 Titolo"), blocco(2, 1, "Testo dell'articolo primo.")]
    report = detect_repeated_headings(indice + corpo)[2]
    assert report["contents_pages"] == [1]
    suppressi = soppressi(indice + corpo)
    assert all(record_key(b) in suppressi for b in indice)
    assert record_key(corpo[0]) not in suppressi


def test_testatina_ripetuta_su_piu_pagine():
    """Stessa intestazione in cima a 3+ pagine: la prima resta, le altre no."""
    records = []
    for page in range(10, 14):
        records.append(blocco(page, 0, "Art. 7 Disposizioni finali"))
        records.append(blocco(page, 1, f"Corpo della pagina {page}."))
    suppressi = soppressi(records)
    intestazioni = [r for r in records if r["page_order"] == 0]
    assert record_key(intestazioni[0]) not in suppressi
    assert all(record_key(r) in suppressi for r in intestazioni[1:])


def test_duplicato_senza_corpo_soppresso():
    """Un rimando non ha testo prima dell'intestazione successiva."""
    records = [
        blocco(1, 0, "Art. 5 Definizioni"),
        blocco(1, 1, "Ai fini del presente atto si applicano le definizioni seguenti."),
        blocco(4, 0, "Art. 5 Definizioni"),   # rimando: subito seguito da altra intestazione
        blocco(4, 1, "Art. 9 Sanzioni"),
        blocco(4, 2, "Chiunque violi le disposizioni e' punito."),
    ]
    suppressi = soppressi(records)
    assert record_key(records[0]) not in suppressi
    assert record_key(records[2]) in suppressi
    assert record_key(records[3]) not in suppressi


def test_duplicato_con_corpo_su_entrambe_le_occorrenze_resta():
    """Numerazione che riparte in allegato: due partizioni diverse, entrambe reali."""
    records = [
        blocco(1, 0, "Art. 1 Oggetto"),
        blocco(1, 1, "Il presente atto disciplina quanto segue."),
        blocco(9, 0, "Art. 1 Oggetto"),
        blocco(9, 1, "Il presente allegato elenca i requisiti."),
    ]
    assert soppressi(records) == set()


def test_partizione_senza_testo_non_sparisce():
    """Se NESSUNA occorrenza ha corpo non si sceglie a caso: restano tutte."""
    records = [
        blocco(1, 0, "Art. 3 Riservato"),
        blocco(5, 0, "Art. 3 Riservato"),
    ]
    # Nessuna delle due ha corpo: la regola non ha un criterio per preferirne una.
    assert detect_repeated_headings(records)[2]["empty_duplicate_headings"] == 0


def test_documento_pulito_non_viene_toccato():
    records = [
        blocco(1, 0, "Art. 1 Oggetto"),
        blocco(1, 1, "Testo primo."),
        blocco(2, 0, "Art. 2 Definizioni"),
        blocco(2, 1, "Testo secondo."),
    ]
    suppressi, downgrade, report = detect_repeated_headings(records)
    assert suppressi == set()
    assert downgrade == set()
    assert report["suppressed_headings"] == 0
    assert report["contents_pages"] == []


def test_il_contenuto_non_e_mai_soppresso():
    """La soppressione riguarda solo le intestazioni, mai il testo normativo."""
    records = [blocco(1, i, "Il titolare del trattamento adotta misure adeguate.") for i in range(8)]
    assert soppressi(records) == set()


def test_ripetizione_con_rubrica_minuscola_declassata_a_content():
    """Un rimando a meta' frase riconosciuto per errore come intestazione:
    stesso numero della vera intestazione, ma la 'rubrica' catturata continua
    in minuscolo invece di essere un titolo autonomo. Va tenuta come testo, non
    soppressa (il contenuto e' normativo vero) e non lasciata aprire un nodo
    duplicato."""
    records = [
        blocco(2, 0, "Art. 5 Definizioni"),
        blocco(2, 1, "Ai fini del presente atto si applicano le definizioni seguenti."),
        blocco(30, 5, "Art. 5 non si applica ai casi previsti al comma precedente."),
        blocco(30, 6, "Il presente comma prosegue con ulteriori disposizioni."),
    ]
    assert record_key(records[2]) in declassati(records)
    assert record_key(records[2]) not in soppressi(records)
    assert record_key(records[0]) not in declassati(records)


def test_duplicato_con_rubrica_titolo_non_declassato():
    """La stessa 'Art. 1 Oggetto' che riparte in un allegato: entrambe le
    rubriche sono titoli autonomi (maiuscola), nessuna va declassata — e' il
    caso gia' coperto da test_duplicato_con_corpo_su_entrambe_le_occorrenze_resta,
    verificato qui anche sul nuovo set."""
    records = [
        blocco(1, 0, "Art. 1 Oggetto"),
        blocco(1, 1, "Il presente atto disciplina quanto segue."),
        blocco(9, 0, "Art. 1 Oggetto"),
        blocco(9, 1, "Il presente allegato elenca i requisiti."),
    ]
    assert declassati(records) == set()
