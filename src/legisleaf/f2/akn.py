"""Albero Akoma Ntoso: conversione dell'albero interno di F2 nello standard.

F2 costruisce un albero "di lavoro" (``struttura_ad_albero.json``) con etichette
della pipeline (``part``, ``article``, ``content``, ``paragraph``...) e una forma
comoda per il grafo e per i validatori: ogni nodo ha ``intestazione``,
``contenuto`` (lista piatta di blocchi) e ``figli``.

Quella forma NON e' Akoma Ntoso. Lo standard OASIS LegalDocML (AKN 3.0) chiede:

* una radice ``akomaNtoso`` con dentro un solo documento (``act``);
* il blocco ``meta/identification`` con la tripletta FRBR Work/Expression/Manifestation;
* la separazione ``preface`` / ``preamble`` / ``body`` / ``attachments``;
* i considerando dentro ``preamble/recitals``, gli allegati dentro ``attachments``
  come documenti autonomi (``doc name="annex"``), non come rami del corpo;
* per ogni contenitore gerarchico ``num`` e ``heading`` come ELEMENTI figli, non
  come stringa unica, e il testo dentro ``content/p`` — oppure ``intro``/``wrapUp``
  quando il contenitore ha a sua volta figli gerarchici (mescolare ``content`` e
  figli e' invalido nello schema);
* un ``eId`` su ogni elemento identificabile, costruito secondo l'Akoma Ntoso
  Naming Convention: ``eId`` del padre + ``__`` + prefisso dell'elemento +
  ``_`` + numero (``chp_IV__art_12__para_3__point_a``).

Questo modulo fa quella conversione. Produce prima un ALBERO (dizionari
annidati, salvato in JSON) e poi lo serializza in XML: l'albero e' l'artefatto
navigabile dagli step a valle, l'XML e' la forma canonica dello standard. I due
non possono divergere perche' il secondo e' generato dal primo.

Nulla qui dipende dalla lingua: prefissi, marcatori di paragrafo e pattern di
intestazione arrivano dal profilo linguistico attivo via ``f1_common``.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Iterable

from legisleaf.common import (
    HEADER_PATTERNS,
    LANGUAGE,
    PARAGRAPH_MARKER_RE,
    POINT_MARKER_RE,
    normalize_legal_text,
)

AKN_NAMESPACE = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"
AKN_SCHEMA_VERSION = "3.0"

# Elementi radice ammessi per un documento AKN: il tipo dichiarato dall'utente
# finisce nell'attributo @name, ma il NOME dell'elemento deve restare uno di
# questi, altrimenti il documento non e' piu' Akoma Ntoso.
AKN_DOCUMENT_ELEMENTS = {
    "act",
    "bill",
    "doc",
    "statement",
    "amendment",
    "judgment",
    "debateReport",
    "officialGazette",
}

# Etichette della pipeline -> elementi gerarchici Akoma Ntoso.
HIERARCHY_ELEMENTS = {
    "part": "part",
    "title": "title",
    "chapter": "chapter",
    "section": "section",
    "article": "article",
}

# Prefissi degli eId (Akoma Ntoso Naming Convention, tabella dei prefissi).
EID_PREFIXES = {
    "part": "part",
    "title": "title",
    "chapter": "chp",
    "section": "sec",
    "article": "art",
    "paragraph": "para",
    "point": "point",
    "recital": "rec",
    "citation": "cit",
    "attachment": "att",
    "hcontainer": "hcontainer",
}

# Elementi che contengono blocchi direttamente (``<recital><p>...``): per questi
# il testo NON va incapsulato ne' in ``<content>`` ne' in ``<intro>``, che lo
# schema ammette solo nei contenitori gerarchici.
BLOCK_CONTAINER_ELEMENTS = {
    "recital",
    "recitals",
    "citation",
    "preface",
    "preamble",
    "conclusions",
    "longTitle",
    "formula",
    "mainBody",
}

# Codici ISO 639-2/B richiesti da ``FRBRlanguage`` (lo standard non accetta i
# codici a due lettere del profilo linguistico).
ISO_639_2 = {
    "it": "ita",
    "en": "eng",
    "fr": "fra",
    "de": "deu",
    "es": "spa",
    "pt": "por",
    "nl": "nld",
    "el": "ell",
}

PREAMBLE_MARKER_RE = LANGUAGE.patterns["preamble_marker"]
RECITAL_NUMBER_RE = LANGUAGE.patterns["recital_number"]
RECITAL_START_RE = LANGUAGE.patterns["recital_section_start"]

# Data segnaposto quando il documento non porta con se' una data di adozione:
# ``FRBRdate/@date`` e' obbligatorio e tipizzato ``xsd:date``, non puo' restare
# vuoto. Il valore e' riconoscibile come "non noto" e viene marcato dal @name.
UNKNOWN_DATE = "1970-01-01"

EID_STYLE = os.getenv("AKN_EID_STYLE", "hierarchical").strip().lower()


# --------------------------------------------------------------------------- #
# Identita' del documento (FRBR)                                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AknIdentity:
    """Metadati FRBR del documento: tutto cio' che finisce in ``meta``."""

    document_id: str
    document_name: str
    country: str = "it"
    doctype: str = "act"
    subtype: str | None = None
    number: str | None = None
    date: str = UNKNOWN_DATE
    date_name: str = "unknown"
    language: str = "ita"
    author_href: str = "/akn/ontology/organization/unknown"
    author_show_as: str = "Autorita' non identificata"
    source_href: str = "/akn/ontology/organization/graphrag-pipeline"
    source_show_as: str = "GraphRAG pipeline"
    # Da dove arrivano i metadati: quale regola del profilo ha riconosciuto il
    # tipo di atto, oppure "default" quando nessuna ha fatto presa.
    metadata_source: str = "default"
    metadata_evidence: str | None = None

    @property
    def work_uri(self) -> str:
        # Lo standard vuole il numero dell'atto come ultimo segmento. Quando non
        # si riesce a leggerlo dal documento si ripiega sull'id della run, che
        # almeno e' univoco; l'id resta comunque in un FRBRalias.
        return f"/akn/{self.country}/{self.doctype}/{self.date}/{_uri_token(self.number or self.document_id)}"

    @property
    def expression_uri(self) -> str:
        return f"{self.work_uri}/{self.language}@"

    @property
    def manifestation_uri(self) -> str:
        return f"{self.expression_uri}.xml"


def _uri_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", (value or "").strip()).strip("-")
    return token or "documento"


DOCUMENT_INCIPIT_CHARS = int(os.getenv("AKN_INCIPIT_CHARS", "1200"))


def document_incipit(root: dict, document_name: str) -> str:
    """Il testo su cui si riconosce il tipo di atto.

    E' l'intestazione del documento: il nome della run piu' il contenuto
    attaccato alla radice dell'albero, cioe' tutto cio' che precede la prima
    partizione ("REGOLAMENTO (UE) 2024/1689 DEL PARLAMENTO EUROPEO E DEL
    CONSIGLIO", "DECRETO LEGISLATIVO 31 marzo 2023, n. 36"). Oltre quella zona
    il testo e' articolato e citerebbe altri atti: cercare la' dentro
    scambierebbe una norma citata per il documento stesso.
    """
    parti = [document_name or ""]
    for item in root.get("contenuto") or []:
        testo = (item.get("testo") or "").strip()
        if testo:
            parti.append(testo)
        if sum(len(p) for p in parti) >= DOCUMENT_INCIPIT_CHARS:
            break
    return normalize_legal_text(" ".join(parti))[:DOCUMENT_INCIPIT_CHARS]


def infer_document_metadata(incipit: str) -> dict[str, Any]:
    """Tipo di atto, paese e autorita' emanante, letti dall'incipit.

    Le regole stanno nel profilo linguistico (``akn_document_types``), non qui:
    "REGOLAMENTO", "DECRETO LEGISLATIVO", "LEGGE" sono parole italiane, e il
    modulo che le contenesse smetterebbe di funzionare al primo documento in
    un'altra lingua. La prima regola che fa presa vince, quindi nel profilo
    vanno ordinate dalla piu' specifica alla piu' generica.
    """
    for pattern, rule in LANGUAGE.document_type_rules():
        match = pattern.search(incipit)
        if not match:
            continue
        found = {
            key: rule[key]
            for key in ("doctype", "subtype", "country", "author_href", "author_show_as")
            if rule.get(key)
        }
        # groupdict() costruisce un dizionario nuovo a ogni chiamata: una volta sola.
        gruppi = match.groupdict()
        number = gruppi.get("number")
        if number:
            found["number"] = number.strip()
        found["metadata_source"] = f"profilo:{rule.get('subtype') or rule.get('doctype') or 'regola'}"
        found["metadata_evidence"] = match.group(0).strip()[:120]
        return found
    return {}


def akn_identity_from_env(document_id: str, document_name: str, inferred: dict[str, Any] | None = None) -> AknIdentity:
    """Identita' FRBR: quanto riconosciuto nel documento, poi gli override d'ambiente.

    L'ordine e' deliberato. Cio' che il documento dichiara di se' (tipo di atto,
    paese, numero, autorita') vince sui default; le variabili ``AKN_*`` vincono
    su tutto, perche' sono una dichiarazione esplicita di chi lancia la
    pipeline. La data resta un segnaposto se nessuno la fornisce: e' obbligatoria
    nello schema e non la si inventa.
    """
    inferred = inferred or {}
    language_code = (os.getenv("LANGUAGE_CODE") or LANGUAGE.code or "it").lower()
    date = os.getenv("AKN_DOCUMENT_DATE", "").strip() or UNKNOWN_DATE

    def scelta(env_var: str, chiave: str, default: str | None) -> str | None:
        valore = os.getenv(env_var, "").strip()
        return valore or inferred.get(chiave) or default

    return AknIdentity(
        document_id=document_id,
        document_name=document_name,
        country=(scelta("AKN_COUNTRY", "country", language_code) or language_code).lower(),
        doctype=scelta("AKN_DOCTYPE", "doctype", "act") or "act",
        subtype=scelta("AKN_SUBTYPE", "subtype", None),
        number=scelta("AKN_NUMBER", "number", None),
        date=date,
        date_name="adoption" if date != UNKNOWN_DATE else "unknown",
        language=ISO_639_2.get(language_code, language_code),
        author_href=scelta("AKN_AUTHOR_HREF", "author_href", "/akn/ontology/organization/unknown"),
        author_show_as=scelta("AKN_AUTHOR_SHOW_AS", "author_show_as", "Autorita' non identificata"),
        metadata_source=inferred.get("metadata_source", "default"),
        metadata_evidence=inferred.get("metadata_evidence"),
    )


def akn_identity_from_document(root: dict, document_id: str, document_name: str) -> AknIdentity:
    """Identita' FRBR ricavata dal documento stesso."""
    return akn_identity_from_env(document_id, document_name, infer_document_metadata(document_incipit(root, document_name)))


# --------------------------------------------------------------------------- #
# Nodi dell'albero AKN                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class AknNode:
    """Un elemento Akoma Ntoso.

    L'ordine dei campi e' l'ordine imposto dallo schema per i contenitori
    gerarchici: ``num``, ``heading``, poi ``intro`` / figli / ``wrapUp`` oppure
    ``content``. ``content`` e ``children`` sono mutuamente esclusivi: quando
    arrivano figli, il testo gia' raccolto scivola in ``intro``.
    """

    element: str
    eid: str | None = None
    attrs: dict[str, str] = field(default_factory=dict)
    num: str | None = None
    heading: str | None = None
    intro: list[dict] = field(default_factory=list)
    children: list["AknNode"] = field(default_factory=list)
    wrap_up: list[dict] = field(default_factory=list)
    content: list[dict] = field(default_factory=list)
    source: dict | None = None

    def add_text(self, block: dict) -> None:
        """Aggiunge un ``<p>``, rispettando il vincolo content/figli."""
        if self.children:
            self.wrap_up.append(block)
        else:
            self.content.append(block)

    def promote_content_to_intro(self) -> None:
        if self.content:
            self.intro.extend(self.content)
            self.content = []

    def to_dict(self) -> dict:
        payload: dict[str, Any] = {"element": self.element}
        if self.eid:
            payload["eId"] = self.eid
        if self.attrs:
            payload["attrs"] = dict(self.attrs)
        if self.num is not None:
            payload["num"] = self.num
        if self.heading is not None:
            payload["heading"] = self.heading
        if self.intro:
            payload["intro"] = self.intro
        if self.children:
            payload["children"] = [child.to_dict() for child in self.children]
        if self.wrap_up:
            payload["wrapUp"] = self.wrap_up
        if self.content:
            payload["content"] = self.content
        if self.source:
            payload["source"] = self.source
        return payload


def _block(text: str, item: dict | None = None) -> dict:
    """Un ``<p>``: testo piu' la provenienza del blocco che lo ha generato."""
    block: dict[str, Any] = {"element": "p", "testo": text}
    if item is not None:
        if item.get("etichetta"):
            block["etichetta_pipeline"] = item["etichetta"]
        if item.get("confidenza") is not None:
            block["confidenza"] = item["confidenza"]
        if item.get("source"):
            block["source"] = item["source"]
    return block


class EidAllocator:
    """Assegna gli eId secondo la naming convention, garantendone l'unicita'.

    Lo standard vuole eId unici nel documento. I numeri estratti dall'OCR non lo
    sono sempre (un "Art. 1" ripetuto in un allegato, un numero non riconosciuto):
    in quel caso il costruttore aggiunge un discriminante numerico invece di
    emettere due elementi con lo stesso identificatore, che renderebbe l'XML
    inutilizzabile per qualsiasi riferimento incrociato.
    """

    def __init__(self) -> None:
        self.used: set[str] = set()
        self.collisions: list[str] = []

    def allocate(self, element: str, number: str | None, parent_eid: str | None, position: int) -> str:
        prefix = EID_PREFIXES.get(element, element)
        token = _eid_token(number) or str(position)
        local = f"{prefix}_{token}"
        base = f"{parent_eid}__{local}" if parent_eid and EID_STYLE != "flat" else local

        candidate = base
        counter = 1
        while candidate in self.used:
            counter += 1
            candidate = f"{base}_{counter}"
        if candidate != base:
            self.collisions.append(candidate)
        self.used.add(candidate)
        return candidate


def _eid_token(number: str | None) -> str:
    """Numero normalizzato per un eId: solo alfanumerici e trattini."""
    token = re.sub(r"[^0-9A-Za-z]+", "-", (number or "").strip()).strip("-")
    return token


# --------------------------------------------------------------------------- #
# Lettura dell'albero interno                                                  #
# --------------------------------------------------------------------------- #
def _node_text(node: dict) -> str:
    return (node.get("intestazione") or "").strip()


def _num_and_heading(node: dict) -> tuple[str | None, str | None]:
    """Separa numero e rubrica dall'intestazione grezza.

    ``"Capo IV Disposizioni finali"`` -> ``("IV", "Disposizioni finali")``.
    Lo standard li vuole in due elementi distinti (``num`` e ``heading``):
    tenerli in una stringa unica e' esattamente cio' che rende l'albero interno
    non conforme.
    """
    label = (node.get("etichetta") or "").lower()
    numero = node.get("numero")
    text = normalize_legal_text(_node_text(node))

    if label == "article":
        return numero, (node.get("rubrica") or None)

    pattern = HEADER_PATTERNS.get(label)
    match = pattern.match(text) if pattern else None
    if not match:
        return numero, None

    end = match.end(1)
    if match.lastindex and match.lastindex >= 2 and match.group(2):
        end = max(end, match.end(2))
    heading = text[end:].strip(" .;:,-–—")
    return numero, heading or None


def _recital_num_and_text(node: dict) -> tuple[str | None, str]:
    """``"(12) Il presente regolamento..."`` -> ``("(12)", "Il presente...")``."""
    text = _node_text(node)
    match = RECITAL_NUMBER_RE.match(text)
    if not match:
        numero = node.get("numero")
        return (f"({numero})" if numero else None), text
    return match.group(0).strip(), text[match.end() :].strip()


def _classify_content_item(item: dict) -> tuple[str, str | None, str]:
    """Riconosce un blocco di contenuto come paragrafo, punto o testo semplice.

    Ritorna ``(genere, num, testo_residuo)``. Il num e' il marcatore cosi' come
    appare nel documento (``"1."``, ``"a)"``): lo standard vuole in ``<num>`` il
    numero cosi' come stampato, non una forma normalizzata.
    """
    text = (item.get("testo") or "").strip()
    label = (item.get("etichetta") or "").lower()

    if label == "paragraph":
        match = PARAGRAPH_MARKER_RE.match(text)
        if match:
            return "paragraph", match.group(0).strip(), text[match.end() :].strip()
        return "paragraph", None, text
    if label == "point":
        match = POINT_MARKER_RE.match(text)
        if match:
            return "point", match.group(0).strip(), text[match.end() :].strip()
        return "point", None, text
    return "text", None, text


def _structure_contents(
    items: Iterable[dict],
    parent_eid: str | None,
    allocator: EidAllocator,
) -> tuple[list[dict], list[AknNode]]:
    """Trasforma la lista piatta ``contenuto`` in ``(intro, figli)``.

    L'albero interno tiene i commi e le lettere come blocchi affiancati; AKN li
    vuole annidati: la lettera ``a)`` e' figlia del comma ``1.`` che la precede,
    non sua sorella. Il testo che precede il primo comma e' l'``intro`` del
    contenitore.
    """
    intro: list[dict] = []
    children: list[AknNode] = []
    open_paragraph: AknNode | None = None
    open_point: AknNode | None = None

    for item in items:
        kind, num, text = _classify_content_item(item)

        if kind == "paragraph":
            node = AknNode("paragraph", num=num, source=item.get("source"))
            node.eid = allocator.allocate("paragraph", num, parent_eid, len(children) + 1)
            if text:
                node.content.append(_block(text, item))
            children.append(node)
            open_paragraph, open_point = node, None
            continue

        if kind == "point":
            container = open_paragraph
            if container is not None:
                container.promote_content_to_intro()
                position = len(container.children) + 1
                node = AknNode("point", num=num, source=item.get("source"))
                node.eid = allocator.allocate("point", num, container.eid, position)
                container.children.append(node)
            else:
                node = AknNode("point", num=num, source=item.get("source"))
                node.eid = allocator.allocate("point", num, parent_eid, len(children) + 1)
                children.append(node)
            if text:
                node.content.append(_block(text, item))
            open_point = node
            continue

        if not text:
            continue
        block = _block(text, item)
        if open_point is not None:
            open_point.add_text(block)
        elif open_paragraph is not None:
            open_paragraph.add_text(block)
        elif children:
            # Non dovrebbe accadere: nell'albero interno il contenuto proprio di
            # un nodo precede sempre i suoi figli. Se accade, il testo finisce in
            # coda all'ultimo figlio invece di essere scartato.
            children[-1].add_text(block)
        else:
            intro.append(block)

    return intro, children


def _build_hierarchy_node(node: dict, parent_eid: str | None, position: int, allocator: EidAllocator) -> AknNode:
    label = (node.get("etichetta") or "").lower()
    element = HIERARCHY_ELEMENTS.get(label, "hcontainer")
    num, heading = _num_and_heading(node)

    akn = AknNode(element, num=num, heading=heading, source=node.get("source"))
    if element == "hcontainer":
        akn.attrs["name"] = label or "container"
    akn.eid = allocator.allocate(label if label in EID_PREFIXES else element, num, parent_eid, position)

    own_intro, own_children = _structure_contents(node.get("contenuto") or [], akn.eid, allocator)
    akn.children.extend(own_children)

    for index, child in enumerate(node.get("figli") or [], start=1):
        akn.children.append(_build_hierarchy_node(child, akn.eid, index + len(own_children), allocator))

    if akn.children:
        akn.intro = own_intro
    else:
        akn.content = own_intro
    return akn


def _build_recital(node: dict, position: int, allocator: EidAllocator) -> AknNode:
    num, text = _recital_num_and_text(node)
    akn = AknNode("recital", num=num, source=node.get("source"))
    # ``rec_1``, non ``recitals__rec_1``: il contenitore ``recitals`` e' unico nel
    # documento e la naming convention non lo interpone nel percorso.
    akn.eid = allocator.allocate("recital", node.get("numero") or _eid_token(num), None, position)
    if text:
        akn.content.append(_block(text))
    for item in node.get("contenuto") or []:
        item_text = (item.get("testo") or "").strip()
        if item_text:
            akn.content.append(_block(item_text, item))
    return akn


def _build_attachment(node: dict, position: int, identity: AknIdentity, allocator: EidAllocator) -> AknNode:
    """Un allegato: documento autonomo dentro ``attachments``.

    Nello standard un allegato non e' un ramo del corpo dell'atto ma un
    documento a se' (``doc name="annex"``) con la propria identita' FRBR,
    riferita al Work padre come componente (``!annex_N``).
    """
    attachment = AknNode("attachment")
    attachment.eid = allocator.allocate("attachment", str(position), None, position)

    component = f"!annex_{position}"
    doc = AknNode("doc", attrs={"name": "annex", "contains": "originalVersion"})
    doc.children.append(_meta_node(identity, component=component, with_references=False))

    title = _node_text(node)
    if title:
        preface = AknNode("preface")
        preface.content.append({"element": "p", "class": "docTitle", "testo": title})
        doc.children.append(preface)

    main_body = AknNode("mainBody")
    body_intro, body_children = _structure_contents(node.get("contenuto") or [], attachment.eid, allocator)
    main_body.children.extend(body_children)
    for index, child in enumerate(node.get("figli") or [], start=1):
        main_body.children.append(
            _build_hierarchy_node(child, attachment.eid, index + len(body_children), allocator)
        )

    # ``mainBody`` ammette blocchi diretti: il testo dell'allegato non ha bisogno
    # di un contenitore fittizio. Non puo' pero' restare vuoto, quindi un
    # allegato senza testo estratto conserva un ``<p>`` vuoto.
    if main_body.children:
        main_body.intro = body_intro
    else:
        main_body.content = body_intro or [{"element": "p", "testo": ""}]

    doc.children.append(main_body)
    # ``<attachment>`` nello schema contiene solo il documento allegato: num e
    # heading dell'allegato vivono nella sua preface, non qui.
    attachment.children.append(doc)
    return attachment


# --------------------------------------------------------------------------- #
# meta / identification                                                        #
# --------------------------------------------------------------------------- #
def _frbr_element(name: str, attrs: dict[str, str]) -> AknNode:
    return AknNode(name, attrs=attrs)


def _meta_node(identity: AknIdentity, *, component: str = "!main", with_references: bool = True) -> AknNode:
    meta = AknNode("meta")
    identification = AknNode("identification", attrs={"source": "#graphrag-pipeline"})

    work = AknNode("FRBRWork")
    work.children.extend(
        [
            _frbr_element("FRBRthis", {"value": f"{identity.work_uri}/{component}"}),
            _frbr_element("FRBRuri", {"value": identity.work_uri}),
            # L'id della run resta agganciato all'atto: e' la chiave con cui il
            # consumatori a valle (grafo, indici, validatori) lo identificano.
            _frbr_element("FRBRalias", {"value": identity.document_id, "name": "pipelineDocumentId"}),
            _frbr_element("FRBRdate", {"date": identity.date, "name": identity.date_name}),
            _frbr_element("FRBRauthor", {"href": "#author"}),
            _frbr_element("FRBRcountry", {"value": identity.country}),
        ]
    )
    # Ordine imposto dallo schema dentro FRBRWork: country, subtype, number.
    if identity.subtype:
        work.children.append(_frbr_element("FRBRsubtype", {"value": identity.subtype}))
    if identity.number:
        work.children.append(_frbr_element("FRBRnumber", {"value": identity.number}))

    expression = AknNode("FRBRExpression")
    expression.children.extend(
        [
            _frbr_element("FRBRthis", {"value": f"{identity.expression_uri}/{component}"}),
            _frbr_element("FRBRuri", {"value": identity.expression_uri}),
            _frbr_element("FRBRdate", {"date": identity.date, "name": identity.date_name}),
            _frbr_element("FRBRauthor", {"href": "#author"}),
            _frbr_element("FRBRlanguage", {"language": identity.language}),
        ]
    )

    manifestation = AknNode("FRBRManifestation")
    manifestation.children.extend(
        [
            _frbr_element("FRBRthis", {"value": f"{identity.manifestation_uri}/{component}"}),
            _frbr_element("FRBRuri", {"value": identity.manifestation_uri}),
            _frbr_element("FRBRdate", {"date": identity.date, "name": identity.date_name}),
            _frbr_element("FRBRauthor", {"href": "#author"}),
            _frbr_element("FRBRformat", {"value": "xml"}),
        ]
    )

    identification.children.extend([work, expression, manifestation])
    meta.children.append(identification)

    # Gli allegati sono documenti dentro lo stesso file: ripetere il blocco
    # ``references`` duplicherebbe gli eId ``graphrag-pipeline`` e ``author``,
    # che devono restare univoci. Il meta dell'allegato si limita a puntarvi
    # tramite ``identification/@source``.
    if not with_references:
        return meta

    references = AknNode("references", attrs={"source": "#graphrag-pipeline"})
    references.children.extend(
        [
            AknNode(
                "TLCOrganization",
                eid="graphrag-pipeline",
                attrs={"href": identity.source_href, "showAs": identity.source_show_as},
            ),
            AknNode(
                "TLCOrganization",
                eid="author",
                attrs={"href": identity.author_href, "showAs": identity.author_show_as},
            ),
        ]
    )
    meta.children.append(references)
    return meta


# --------------------------------------------------------------------------- #
# Costruzione del documento                                                    #
# --------------------------------------------------------------------------- #
def build_akn_tree(root: dict, identity: AknIdentity) -> dict:
    """Albero Akoma Ntoso a partire dalla radice virtuale dell'albero interno."""
    allocator = EidAllocator()

    recital_nodes: list[dict] = []
    annex_nodes: list[dict] = []
    body_nodes: list[dict] = []
    for child in root.get("figli") or []:
        label = (child.get("etichetta") or "").lower()
        if label == "recital":
            recital_nodes.append(child)
        elif label == "annex":
            annex_nodes.append(child)
        else:
            body_nodes.append(child)

    # Il contenuto attaccato alla radice e' cio' che precede la prima partizione.
    # Nello standard non e' un blocco unico: i "visto/vista" sono ``citation`` del
    # preambolo, la formula che apre i considerando e' l'``intro`` di
    # ``recitals``, il resto (titolo, intestazione dell'atto) e' preface.
    citations: list[dict] = []
    recitals_intro: list[dict] = []
    preface_blocks: list[dict] = []
    for item in root.get("contenuto") or []:
        text = (item.get("testo") or "").strip()
        if not text:
            continue
        normalized = normalize_legal_text(text)
        if RECITAL_START_RE.match(normalized):
            recitals_intro.append(item)
        elif PREAMBLE_MARKER_RE.match(normalized) and not RECITAL_NUMBER_RE.match(text):
            citations.append(item)
        else:
            preface_blocks.append(item)

    # Senza partizioni riconosciute il testo libero della radice e' l'unico corpo
    # che il documento ha: metterlo nella preface lascerebbe il body vuoto (e
    # duplicherebbe il testo se il body dovesse poi ripescarlo).
    fallback_body_blocks: list[dict] = []
    if not body_nodes:
        fallback_body_blocks, preface_blocks = preface_blocks, []

    element_name = identity.doctype if identity.doctype in AKN_DOCUMENT_ELEMENTS else "act"
    document = AknNode(
        element_name,
        attrs={"name": identity.doctype, "contains": "originalVersion"},
    )
    document.children.append(_meta_node(identity))

    preface = AknNode("preface")
    long_title = AknNode("longTitle")
    long_title.content.append({"element": "p", "testo": identity.document_name})
    preface.children.append(long_title)
    # I ``<p>`` della preface vanno DOPO il longTitle: nel serializzatore il
    # blocco ``content`` e' emesso dopo i figli, quindi l'ordine e' corretto.
    for item in preface_blocks:
        preface.content.append(_block((item.get("testo") or "").strip(), item))
    document.children.append(preface)

    if citations or recital_nodes or recitals_intro:
        preamble = AknNode("preamble")
        for index, item in enumerate(citations, start=1):
            citation = AknNode("citation")
            citation.eid = allocator.allocate("citation", str(index), None, index)
            citation.content.append(_block((item.get("testo") or "").strip(), item))
            preamble.children.append(citation)
        if recital_nodes or recitals_intro:
            recitals = AknNode("recitals", eid="recitals")
            recitals.intro = [_block((item.get("testo") or "").strip(), item) for item in recitals_intro]
            for index, node in enumerate(recital_nodes, start=1):
                recitals.children.append(_build_recital(node, index, allocator))
            preamble.children.append(recitals)
        document.children.append(preamble)

    body = AknNode("body")
    for index, node in enumerate(body_nodes, start=1):
        body.children.append(_build_hierarchy_node(node, None, index, allocator))
    if not body.children:
        # ``body`` deve contenere almeno un elemento gerarchico: un documento
        # senza partizioni riconosciute finisce in un contenitore generico
        # invece di produrre un corpo vuoto (invalido).
        wrapper = AknNode("hcontainer", attrs={"name": "content"})
        wrapper.eid = allocator.allocate("hcontainer", "1", None, 1)
        intro, wrapper_children = _structure_contents(fallback_body_blocks, wrapper.eid, allocator)
        wrapper.children = wrapper_children
        if wrapper_children:
            wrapper.intro = intro
        else:
            wrapper.content = intro or [{"element": "p", "testo": ""}]
        body.children.append(wrapper)
    document.children.append(body)

    if annex_nodes:
        attachments = AknNode("attachments")
        for index, node in enumerate(annex_nodes, start=1):
            attachments.children.append(_build_attachment(node, index, identity, allocator))
        document.children.append(attachments)

    akn = {
        "element": "akomaNtoso",
        "attrs": {"xmlns": AKN_NAMESPACE},
        "akn_version": AKN_SCHEMA_VERSION,
        "eid_style": EID_STYLE,
        "identification": {
            "work": identity.work_uri,
            "expression": identity.expression_uri,
            "manifestation": identity.manifestation_uri,
            "document_id": identity.document_id,
            "document_name": identity.document_name,
            "language": identity.language,
            "country": identity.country,
            "doctype": identity.doctype,
            "subtype": identity.subtype,
            "number": identity.number,
            "date": identity.date,
            "date_name": identity.date_name,
            # Tracciabilita' del riconoscimento: quale regola del profilo ha
            # fatto presa e su quale frammento di testo.
            "metadata_source": identity.metadata_source,
            "metadata_evidence": identity.metadata_evidence,
        },
        "children": [document.to_dict()],
    }
    akn["eid_collisions"] = list(allocator.collisions)
    return akn


# --------------------------------------------------------------------------- #
# Serializzazione XML                                                          #
# --------------------------------------------------------------------------- #
def _qname(tag: str) -> str:
    return f"{{{AKN_NAMESPACE}}}{tag}"


def _append_blocks(parent: ET.Element, blocks: list[dict]) -> None:
    for block in blocks:
        element = ET.SubElement(parent, _qname(block.get("element", "p")))
        if block.get("class"):
            element.set("class", block["class"])
        element.text = block.get("testo") or ""


def _append_node(parent: ET.Element, node: dict) -> None:
    element = ET.SubElement(parent, _qname(node["element"]))
    for key, value in (node.get("attrs") or {}).items():
        if key == "xmlns":
            continue
        element.set(key, str(value))
    if node.get("eId"):
        element.set("eId", node["eId"])

    if node.get("num") is not None:
        ET.SubElement(element, _qname("num")).text = node["num"]
    if node.get("heading") is not None:
        ET.SubElement(element, _qname("heading")).text = node["heading"]

    if node.get("intro"):
        if node["element"] in BLOCK_CONTAINER_ELEMENTS:
            _append_blocks(element, node["intro"])
        else:
            intro = ET.SubElement(element, _qname("intro"))
            _append_blocks(intro, node["intro"])

    for child in node.get("children") or []:
        _append_node(element, child)

    if node.get("wrapUp"):
        wrap_up = ET.SubElement(element, _qname("wrapUp"))
        _append_blocks(wrap_up, node["wrapUp"])

    # Il testo di un elemento identificabile sta sempre in blocchi ``<p>``, mai
    # come testo diretto: ``_append_blocks`` e' l'unico punto che scrive
    # ``element.text``, e i blocchi arrivano solo da intro/wrapUp/content.
    if node.get("content"):
        if node["element"] in BLOCK_CONTAINER_ELEMENTS:
            _append_blocks(element, node["content"])
        else:
            content = ET.SubElement(element, _qname("content"))
            _append_blocks(content, node["content"])


def akn_tree_to_xml(akn: dict) -> bytes:
    """Serializza l'albero AKN nell'XML dello standard (namespace di default)."""
    ET.register_namespace("", AKN_NAMESPACE)
    root = ET.Element(_qname("akomaNtoso"))
    for child in akn.get("children") or []:
        _append_node(root, child)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


# --------------------------------------------------------------------------- #
# Controlli sull'albero prodotto                                               #
# --------------------------------------------------------------------------- #
def check_akn_tree(akn: dict) -> dict:
    """Verifica i vincoli dello standard che il costruttore deve garantire.

    Non sostituisce la validazione contro l'XSD (facoltativa, vedi
    ``validate_against_xsd``): controlla le tre invarianti che, se violate,
    producono un XML sintatticamente valido ma non conforme — eId duplicati,
    ``content`` insieme a figli gerarchici, contenitori obbligatori vuoti.
    """
    report: dict[str, Any] = {
        "akn_version": akn.get("akn_version"),
        "eid_style": akn.get("eid_style"),
        "n_elementi": 0,
        "n_eid": 0,
        "conteggio_elementi": {},
        "eid_duplicati": [],
        "eid_collisioni_risolte": list(akn.get("eid_collisions") or []),
        "content_con_figli": [],
        "contenitori_vuoti": [],
    }
    seen: set[str] = set()

    def walk(node: dict) -> None:
        report["n_elementi"] += 1
        element = node.get("element", "?")
        report["conteggio_elementi"][element] = report["conteggio_elementi"].get(element, 0) + 1

        eid = node.get("eId")
        if eid:
            report["n_eid"] += 1
            if eid in seen:
                report["eid_duplicati"].append(eid)
            seen.add(eid)

        # Vincolo dei soli contenitori gerarchici: ``<content>`` e figli
        # gerarchici si escludono. Nei contenitori di blocchi (preface,
        # recital...) i ``<p>`` convivono con gli altri figli senza wrapper.
        if node.get("content") and node.get("children") and element not in BLOCK_CONTAINER_ELEMENTS:
            report["content_con_figli"].append(eid or element)

        # ``body`` vuole almeno un elemento gerarchico; ``mainBody`` accetta
        # anche blocchi diretti. In entrambi i casi vuoto = documento invalido.
        if element == "body" and not node.get("children"):
            report["contenitori_vuoti"].append(element)
        if element == "mainBody" and not (node.get("children") or node.get("content")):
            report["contenitori_vuoti"].append(element)

        for child in node.get("children") or []:
            walk(child)

    for child in akn.get("children") or []:
        walk(child)

    report["valido"] = not (
        report["eid_duplicati"] or report["content_con_figli"] or report["contenitori_vuoti"]
    )
    return report


def validate_against_xsd(xml_bytes: bytes, xsd_path: str | None = None) -> dict:
    """Validazione contro lo schema ufficiale, se disponibile in locale.

    Lo XSD di Akoma Ntoso non e' distribuito con il repository e non viene
    scaricato: se ``AKN_XSD_PATH`` non punta a un file, la validazione risulta
    semplicemente "non eseguita" e il resto della pipeline non cambia.
    """
    path = xsd_path or os.getenv("AKN_XSD_PATH", "").strip()
    if not path or not os.path.exists(path):
        return {"eseguita": False, "motivo": "XSD non disponibile (AKN_XSD_PATH non impostato)"}
    try:
        from lxml import etree
    except ImportError:
        return {"eseguita": False, "motivo": "lxml non installato"}

    try:
        schema = etree.XMLSchema(etree.parse(path))
    except etree.LxmlError as exc:
        return {"eseguita": False, "motivo": f"XSD non caricabile: {exc}"}

    try:
        document = etree.fromstring(xml_bytes)
    except etree.XMLSyntaxError as exc:
        return {"eseguita": True, "valido": False, "xsd": path, "errori": [str(exc)]}

    if schema.validate(document):
        return {"eseguita": True, "valido": True, "xsd": path}
    return {
        "eseguita": True,
        "valido": False,
        "xsd": path,
        "errori": [str(error) for error in schema.error_log][:50],
    }
