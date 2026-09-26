"""Test della conversione albero interno -> Akoma Ntoso (F2).

Coprono i vincoli dello standard che un serializzatore ingenuo viola in
silenzio, producendo un XML ben formato ma non conforme: ``content`` insieme a
figli gerarchici, ``num``/``heading`` lasciati in un'unica stringa, allegati
appesi al corpo dell'atto, considerando fuori dal preambolo, eId non univoci o
non annidati.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from legisleaf.f2.akn import (
    AKN_NAMESPACE,
    AknIdentity,
    akn_identity_from_document,
    akn_tree_to_xml,
    build_akn_tree,
    check_akn_tree,
    document_incipit,
    infer_document_metadata,
)

NS = {"akn": AKN_NAMESPACE}


def identity() -> AknIdentity:
    return AknIdentity(document_id="atto_prova", document_name="Atto di prova", date="2024-01-01")


def nodo(etichetta: str, intestazione: str, numero: str | None = None, **extra) -> dict:
    node = {
        "name": f"{etichetta}:{numero}" if numero else etichetta,
        "intestazione": intestazione,
        "etichetta": etichetta,
        "numero": numero,
        "confidenza": None,
        "source": {"ordine": 0},
        "contenuto": [],
        "figli": [],
    }
    node.update(extra)
    return node


def contenuto(etichetta: str, testo: str) -> dict:
    return {"etichetta": etichetta, "testo": testo, "confidenza": None, "source": {"ordine": 0}}


@pytest.fixture()
def albero_completo() -> dict:
    articolo = nodo("article", "Art. 1 Oggetto", "1", rubrica="Oggetto")
    articolo["contenuto"] = [
        contenuto("content", "Il presente atto disciplina quanto segue."),
        contenuto("paragraph", "1. Le disposizioni si applicano ai soggetti seguenti:"),
        contenuto("point", "a) le imprese;"),
        contenuto("point", "b) le amministrazioni."),
        contenuto("paragraph", "2. Restano ferme le competenze regionali."),
    ]
    capo = nodo("chapter", "Capo I Disposizioni generali", "I")
    capo["contenuto"] = [contenuto("content", "Il presente capo introduce le definizioni.")]
    capo["figli"] = [articolo]

    allegato = nodo("annex", "Allegato I Elenco dei requisiti", "I")
    allegato["contenuto"] = [contenuto("content", "Requisiti minimi.")]

    recital = nodo("recital", "(1) Il presente atto persegue la finalita' indicata.", "1")

    root = nodo("document", "Atto di prova")
    root["contenuto"] = [
        contenuto("content", "IL PARLAMENTO E IL CONSIGLIO"),
        contenuto("content", "Visto il trattato sul funzionamento dell'Unione europea,"),
        contenuto("content", "considerando quanto segue:"),
    ]
    root["figli"] = [recital, capo, allegato]
    return root


@pytest.fixture()
def xml_completo(albero_completo) -> ET.Element:
    return ET.fromstring(akn_tree_to_xml(build_akn_tree(albero_completo, identity())))


# --------------------------------------------------------------------------- #
# Struttura del documento                                                      #
# --------------------------------------------------------------------------- #
def test_radice_e_namespace(xml_completo):
    assert xml_completo.tag == f"{{{AKN_NAMESPACE}}}akomaNtoso"
    assert xml_completo.find("akn:act", NS) is not None


def test_sezioni_nell_ordine_dello_standard(xml_completo):
    act = xml_completo.find("akn:act", NS)
    sezioni = [child.tag.split("}")[1] for child in act]
    assert sezioni == ["meta", "preface", "preamble", "body", "attachments"]


def test_identificazione_frbr_completa(xml_completo):
    identification = xml_completo.find("akn:act/akn:meta/akn:identification", NS)
    assert [child.tag.split("}")[1] for child in identification] == [
        "FRBRWork",
        "FRBRExpression",
        "FRBRManifestation",
    ]
    assert identification.find("akn:FRBRWork/akn:FRBRuri", NS).get("value") == "/akn/it/act/2024-01-01/atto_prova"
    assert identification.find("akn:FRBRExpression/akn:FRBRlanguage", NS).get("language") == "ita"


def test_considerando_nel_preambolo_non_nel_corpo(xml_completo):
    recital = xml_completo.find("akn:act/akn:preamble/akn:recitals/akn:recital", NS)
    assert recital is not None
    assert recital.get("eId") == "rec_1"
    assert recital.find("akn:num", NS).text == "(1)"
    # Il testo di un recital sta in <p> diretti: <content> non e' ammesso.
    assert recital.find("akn:content", NS) is None
    assert recital.find("akn:p", NS).text.startswith("Il presente atto persegue")
    assert xml_completo.find("akn:act/akn:body//akn:recital", NS) is None


def test_visto_diventa_citazione_del_preambolo(xml_completo):
    citazioni = xml_completo.findall("akn:act/akn:preamble/akn:citation", NS)
    assert [c.find("akn:p", NS).text for c in citazioni] == [
        "Visto il trattato sul funzionamento dell'Unione europea,"
    ]
    # Il resto del testo di radice resta nella preface.
    preface = xml_completo.find("akn:act/akn:preface", NS)
    assert preface.find("akn:longTitle/akn:p", NS).text == "Atto di prova"
    assert [p.text for p in preface.findall("akn:p", NS)] == ["IL PARLAMENTO E IL CONSIGLIO"]


def test_formula_dei_considerando_e_intro_dei_recitals(xml_completo):
    """"considerando quanto segue:" apre i recitals, non e' una citazione."""
    recitals = xml_completo.find("akn:act/akn:preamble/akn:recitals", NS)
    assert [p.text for p in recitals.findall("akn:p", NS)] == ["considerando quanto segue:"]
    # ``intro`` e' ammesso solo nei contenitori gerarchici.
    assert recitals.find("akn:intro", NS) is None


def test_allegato_in_attachments_come_documento_autonomo(xml_completo):
    attachment = xml_completo.find("akn:act/akn:attachments/akn:attachment", NS)
    assert attachment.get("eId") == "att_1"
    doc = attachment.find("akn:doc", NS)
    assert doc.get("name") == "annex"
    assert doc.find("akn:meta/akn:identification", NS) is not None
    assert doc.find("akn:preface/akn:p", NS).text == "Allegato I Elenco dei requisiti"
    main_body = doc.find("akn:mainBody", NS)
    # ``mainBody`` accetta blocchi diretti ma non ``intro``, che lo schema
    # riserva ai contenitori gerarchici.
    assert main_body.find("akn:intro", NS) is None
    assert main_body.find("akn:p", NS).text == "Requisiti minimi."
    assert xml_completo.find("akn:act/akn:body//akn:doc", NS) is None


# --------------------------------------------------------------------------- #
# Gerarchia, num/heading, eId                                                  #
# --------------------------------------------------------------------------- #
def test_num_e_heading_sono_elementi_distinti(xml_completo):
    capo = xml_completo.find("akn:act/akn:body/akn:chapter", NS)
    assert capo.find("akn:num", NS).text == "I"
    assert capo.find("akn:heading", NS).text == "Disposizioni generali"

    articolo = capo.find("akn:article", NS)
    assert articolo.find("akn:num", NS).text == "1"
    assert articolo.find("akn:heading", NS).text == "Oggetto"


def test_eid_annidati_secondo_la_naming_convention(xml_completo):
    capo = xml_completo.find("akn:act/akn:body/akn:chapter", NS)
    assert capo.get("eId") == "chp_I"
    articolo = capo.find("akn:article", NS)
    assert articolo.get("eId") == "chp_I__art_1"
    comma = articolo.find("akn:paragraph", NS)
    assert comma.get("eId") == "chp_I__art_1__para_1"
    assert [p.get("eId") for p in comma.findall("akn:point", NS)] == [
        "chp_I__art_1__para_1__point_a",
        "chp_I__art_1__para_1__point_b",
    ]


def test_lettere_annidate_nel_comma_che_le_introduce(xml_completo):
    comma = xml_completo.find("akn:act/akn:body/akn:chapter/akn:article/akn:paragraph", NS)
    assert comma.find("akn:num", NS).text == "1."
    # Il comma ha figli: il suo testo diventa <intro>, non <content>.
    assert comma.find("akn:intro/akn:p", NS).text == "Le disposizioni si applicano ai soggetti seguenti:"
    assert comma.find("akn:content", NS) is None
    punto = comma.find("akn:point", NS)
    assert punto.find("akn:num", NS).text == "a)"
    assert punto.find("akn:content/akn:p", NS).text == "le imprese;"


def test_testo_prima_del_primo_comma_diventa_intro_dell_articolo(xml_completo):
    articolo = xml_completo.find("akn:act/akn:body/akn:chapter/akn:article", NS)
    assert articolo.find("akn:intro/akn:p", NS).text == "Il presente atto disciplina quanto segue."
    assert articolo.find("akn:content", NS) is None
    assert len(articolo.findall("akn:paragraph", NS)) == 2


def test_contenitore_senza_figli_usa_content(xml_completo):
    capo = xml_completo.find("akn:act/akn:body/akn:chapter", NS)
    # Il capo ha figli -> intro; l'ultimo comma non ne ha -> content.
    assert capo.find("akn:intro/akn:p", NS).text == "Il presente capo introduce le definizioni."
    ultimo_comma = capo.findall("akn:article/akn:paragraph", NS)[-1]
    assert ultimo_comma.find("akn:content/akn:p", NS).text == "Restano ferme le competenze regionali."


# --------------------------------------------------------------------------- #
# Invarianti verificate dal report                                             #
# --------------------------------------------------------------------------- #
def test_report_conferma_la_conformita(albero_completo):
    report = check_akn_tree(build_akn_tree(albero_completo, identity()))
    assert report["valido"]
    assert report["eid_duplicati"] == []
    assert report["content_con_figli"] == []
    assert report["contenitori_vuoti"] == []
    assert report["conteggio_elementi"]["article"] == 1
    assert report["conteggio_elementi"]["point"] == 2


def test_numeri_ripetuti_non_producono_eid_duplicati():
    """Due "Art. 1" nello stesso ramo: lo standard vuole eId univoci."""
    root = nodo("document", "Atto")
    root["figli"] = [nodo("article", "Art. 1 Primo", "1"), nodo("article", "Art. 1 Secondo", "1")]
    akn = build_akn_tree(root, identity())
    report = check_akn_tree(akn)
    assert report["eid_duplicati"] == []
    assert report["eid_collisioni_risolte"] == ["art_1_2"]


def test_documento_senza_partizioni_non_lascia_il_corpo_vuoto():
    root = nodo("document", "Atto")
    root["contenuto"] = [contenuto("content", "Testo libero senza partizioni.")]
    akn = build_akn_tree(root, identity())
    assert check_akn_tree(akn)["valido"]
    xml = ET.fromstring(akn_tree_to_xml(akn))
    contenitore = xml.find("akn:act/akn:body/akn:hcontainer", NS)
    assert contenitore.get("name") == "content"
    assert contenitore.find("akn:content/akn:p", NS).text == "Testo libero senza partizioni."
    # Il testo non deve comparire due volte (corpo e preface).
    assert xml.findall("akn:act/akn:preface/akn:p", NS) == []


# --------------------------------------------------------------------------- #
# Metadati ricavati dal documento                                              #
# --------------------------------------------------------------------------- #
def albero_con_incipit(*righe: str) -> dict:
    root = nodo("document", "documento")
    root["contenuto"] = [contenuto("content", riga) for riga in righe]
    root["figli"] = [nodo("article", "Art. 1 Oggetto", "1", rubrica="Oggetto")]
    return root


@pytest.mark.parametrize(
    "incipit, subtype, country, number",
    [
        (
            "REGOLAMENTO (UE) 2024/1689 DEL PARLAMENTO EUROPEO E DEL CONSIGLIO",
            "regolamento",
            "eu",
            "2024/1689",
        ),
        ("DIRETTIVA (UE) 2022/2555 DEL PARLAMENTO EUROPEO E DEL CONSIGLIO", "direttiva", "eu", "2022/2555"),
        ("DECRETO LEGISLATIVO 31 marzo 2023, n. 36", "decreto-legislativo", "it", "36"),
        ("DECRETO-LEGGE 2 marzo 2024, n. 19", "decreto-legge", "it", "19"),
        ("LEGGE 5 agosto 2022, n. 118", "legge", "it", "118"),
        ("COSTITUZIONE DELLA REPUBBLICA ITALIANA", "costituzione", "it", None),
    ],
)
def test_tipo_di_atto_riconosciuto_dall_incipit(incipit, subtype, country, number):
    identity = akn_identity_from_document(albero_con_incipit(incipit), "doc", "doc")
    assert identity.subtype == subtype
    assert identity.country == country
    assert identity.number == number
    assert identity.metadata_source == f"profilo:{subtype}"


def test_autorita_emanante_dal_tipo_di_atto():
    identity = akn_identity_from_document(
        albero_con_incipit("REGOLAMENTO (UE) 2024/1689 DEL PARLAMENTO EUROPEO E DEL CONSIGLIO"), "ai_act", "ai_act"
    )
    assert identity.author_show_as == "Parlamento europeo e Consiglio dell'Unione europea"
    assert identity.work_uri.startswith("/akn/eu/act/")
    assert identity.work_uri.endswith("/2024-1689")


def test_documento_non_riconosciuto_resta_sui_default():
    identity = akn_identity_from_document(albero_con_incipit("Testo senza intestazione riconoscibile"), "x", "x")
    assert identity.metadata_source == "default"
    assert identity.subtype is None
    assert identity.country == "it"
    assert identity.number is None
    assert identity.work_uri.endswith("/x")


def test_variabile_d_ambiente_vince_sul_riconoscimento(monkeypatch):
    root = albero_con_incipit("REGOLAMENTO (UE) 2024/1689 DEL PARLAMENTO EUROPEO E DEL CONSIGLIO")
    monkeypatch.setenv("AKN_COUNTRY", "it")
    monkeypatch.setenv("AKN_SUBTYPE", "regolamento-delegato")
    identity = akn_identity_from_document(root, "doc", "doc")
    assert identity.country == "it"
    assert identity.subtype == "regolamento-delegato"
    # Il numero non e' stato sovrascritto: resta quello letto nel documento.
    assert identity.number == "2024/1689"


def test_incipit_si_ferma_prima_dell_articolato():
    """Un atto citato in un articolo non deve diventare il tipo del documento."""
    root = albero_con_incipit("DECRETO LEGISLATIVO 31 marzo 2023, n. 36")
    root["figli"][0]["contenuto"] = [
        contenuto("content", "Si applica il REGOLAMENTO (UE) 2016/679 del Parlamento europeo.")
    ]
    incipit = document_incipit(root, "codice_contratti")
    assert "REGOLAMENTO (UE) 2016/679" not in incipit
    assert infer_document_metadata(incipit)["subtype"] == "decreto-legislativo"


def test_metadati_riconosciuti_finiscono_nel_frbr():
    root = albero_con_incipit("DECRETO LEGISLATIVO 31 marzo 2023, n. 36")
    identity = akn_identity_from_document(root, "codice_contratti", "codice_contratti")
    xml = ET.fromstring(akn_tree_to_xml(build_akn_tree(root, identity)))
    work = xml.find("akn:act/akn:meta/akn:identification/akn:FRBRWork", NS)
    assert work.find("akn:FRBRcountry", NS).get("value") == "it"
    assert work.find("akn:FRBRsubtype", NS).get("value") == "decreto-legislativo"
    assert work.find("akn:FRBRnumber", NS).get("value") == "36"
    # L'id della run resta agganciato all'atto anche se l'URI usa il numero.
    assert work.find("akn:FRBRalias", NS).get("value") == "codice_contratti"
    assert [child.tag.split("}")[1] for child in work] == [
        "FRBRthis",
        "FRBRuri",
        "FRBRalias",
        "FRBRdate",
        "FRBRauthor",
        "FRBRcountry",
        "FRBRsubtype",
        "FRBRnumber",
    ]


def test_albero_json_e_xml_descrivono_la_stessa_struttura(albero_completo):
    """L'XML e' generato dall'albero JSON: gli eId devono coincidere uno a uno."""
    akn = build_akn_tree(albero_completo, identity())
    xml = ET.fromstring(akn_tree_to_xml(akn))

    def eid_albero(node: dict) -> list[str]:
        found = [node["eId"]] if node.get("eId") else []
        for child in node.get("children") or []:
            found.extend(eid_albero(child))
        return found

    attesi = [e for child in akn["children"] for e in eid_albero(child)]
    trovati = [el.get("eId") for el in xml.iter() if el.get("eId")]
    assert attesi == trovati
