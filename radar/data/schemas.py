"""Domain schemas shared by the generator, the database, the tools and the agent."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Topic(str, Enum):
    PFAS = "pfas"
    ROHS = "rohs"
    REACH_SVHC = "reach_svhc"
    CONFLICT_MINERALS = "conflict_minerals"
    RBA_LABOUR = "rba_labour"
    EXPORT_CONTROL = "export_control"
    CSRD_ESG = "csrd_esg"
    BATTERY = "battery"
    PACKAGING = "packaging"
    CYBERSECURITY = "cybersecurity"
    DATA_PROTECTION = "data_protection"
    SUPPLIER_INCIDENT = "supplier_incident"
    OTHER = "other"


class Jurisdiction(str, Enum):
    EU = "EU"
    US = "US"
    TW = "TW"
    CN = "CN"
    GLOBAL = "GLOBAL"


# Substances/materials that regulations talk about, mapped to the topic they trigger.
MATERIAL_TOPIC: dict[str, Topic] = {
    "PTFE coating": Topic.PFAS, "fluoropolymer gasket": Topic.PFAS, "PFAS surfactant": Topic.PFAS,
    "lead solder": Topic.ROHS, "cadmium plating": Topic.ROHS, "hexavalent chromium": Topic.ROHS,
    "DEHP plasticizer": Topic.REACH_SVHC, "brominated flame retardant": Topic.REACH_SVHC, "bisphenol A": Topic.REACH_SVHC,
    "tantalum capacitor": Topic.CONFLICT_MINERALS, "tin plating": Topic.CONFLICT_MINERALS, "tungsten contact": Topic.CONFLICT_MINERALS, "cobalt cathode": Topic.CONFLICT_MINERALS,
    "lithium-ion cell": Topic.BATTERY, "nickel-metal hydride cell": Topic.BATTERY,
    "EPS foam packaging": Topic.PACKAGING, "PVC blister": Topic.PACKAGING,
    "high-performance FPGA": Topic.EXPORT_CONTROL, "GaN RF amplifier": Topic.EXPORT_CONTROL, "encryption module": Topic.EXPORT_CONTROL,
    "aluminium housing": Topic.OTHER, "copper busbar": Topic.OTHER, "silicon MOSFET": Topic.OTHER, "ceramic capacitor": Topic.OTHER,
    "steel bracket": Topic.OTHER, "ABS enclosure": Topic.OTHER, "glass-fibre PCB": Topic.OTHER,
}


class Supplier(BaseModel):
    supplier_id: str
    name: str
    country: str
    tier: int = Field(ge=1, le=3)
    certifications: list[str]
    materials: list[str]
    rba_audit_score: int = Field(ge=0, le=200)
    risk_flags: list[str] = []
    contact_email: str


class Part(BaseModel):
    part_id: str
    description: str
    material: str
    supplier_id: str
    unit_cost_usd: float


class Product(BaseModel):
    product_id: str
    name: str
    markets: list[str]
    bom: list[tuple[str, int]]  # (part_id, qty)


class DocChunkMeta(BaseModel):
    doc_id: str
    doc_type: Literal["policy", "contract", "sop", "regulation"]
    title: str
    section: str
    topics: list[str]
    jurisdiction: str = "GLOBAL"
    supplier_id: str | None = None
    language: str = "en"
    effective_date: str = ""


class FeedEvent(BaseModel):
    event_id: str
    source: str
    url: str
    title: str
    summary: str
    published: datetime
    jurisdiction: Jurisdiction = Jurisdiction.GLOBAL
    topic: Topic = Topic.OTHER
    urgency: Literal["low", "medium", "high"] = "medium"
    language: str = "en"
    raw: dict = {}


class Citation(BaseModel):
    doc_id: str = Field(description="Document id exactly as in the evidence, e.g. POL-ENV-001 or CTR-S0010")
    section: str = Field(description="Section heading as in the evidence, e.g. '4 Response to regulatory change'")
    quote: str = Field(default="", max_length=300, description="Short verbatim quote from that section")


class AffectedItem(BaseModel):
    kind: Literal["supplier", "part", "product", "contract_clause", "policy"]
    id: str = Field(description="Exact id: supplier_id (S0001), part_id (P00001), product_id (PSU-1200), or the document id for a contract clause or policy (CTR-S0010, POL-ENV-001)")
    name: str
    why: str = Field(max_length=300, description="Why this item is affected by the event")
    citations: list[Citation] = []


class RequiredAction(BaseModel):
    action: str
    owner_role: str
    deadline: str = Field(description="Time limit as stated in the cited passage, or 'not specified in policy'. Never a made-up calendar date.")
    citations: list[Citation] = []


class BriefDraft(BaseModel):
    """What the model writes. Deliberately contains no ids or numbers the pipeline owns."""

    title: str
    summary: str = Field(max_length=1200)
    topic: Topic
    jurisdiction: Jurisdiction
    affected: list[AffectedItem]
    actions: list[RequiredAction]
    confidence: float = Field(ge=0, le=1)
    open_questions: list[str] = []
    citations: list[Citation]


class ImpactBrief(BriefDraft):
    """The stored brief: the model's draft plus fields only the pipeline may set."""

    event_id: str
    requires_approval: bool = True
    # Totals from SQL behind the "top items" the model chose to describe; never written by the model.
    affected_counts: dict[str, int] = {}


class SubQuestions(BaseModel):
    sub_questions: list[str] = Field(min_length=1, max_length=5)


class CriticVerdict(BaseModel):
    approved: bool
    missing_citations: list[str] = []
    notes: str = ""
