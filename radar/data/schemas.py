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
    doc_id: str
    section: str
    quote: str = Field(max_length=300)


class AffectedItem(BaseModel):
    kind: Literal["supplier", "part", "product", "contract_clause", "policy"]
    id: str
    name: str
    why: str = Field(max_length=300)
    citations: list[Citation] = []


class RequiredAction(BaseModel):
    action: str
    owner_role: str
    deadline: str
    citations: list[Citation] = []


class ImpactBrief(BaseModel):
    event_id: str
    title: str
    summary: str = Field(max_length=1200)
    topic: Topic
    jurisdiction: Jurisdiction
    affected: list[AffectedItem]
    actions: list[RequiredAction]
    confidence: float = Field(ge=0, le=1)
    open_questions: list[str] = []
    citations: list[Citation]
    requires_approval: bool = True
