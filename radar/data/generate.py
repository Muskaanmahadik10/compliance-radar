"""Deterministic synthetic company: an electronics manufacturer with 200 suppliers, ~600 parts, 20 products, 40 policies
and 30 supplier contracts. Every regulated material is linked to suppliers, parts, products and the policy/contract
clauses that mention it, which gives the evaluation its ground truth ("a PFAS restriction affects exactly these items").

Usage: python -m radar.data.generate [--seed 7]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from radar.config import settings
from radar.data.schemas import MATERIAL_TOPIC, Part, Product, Supplier, Topic

COUNTRIES = ["Taiwan", "China", "Vietnam", "Malaysia", "Japan", "South Korea", "Thailand", "Germany", "United States", "Mexico", "India", "Philippines"]
NAME_A = ["Everbright", "Golden", "Pacific", "Harmony", "Summit", "Unity", "Dragon", "Lotus", "Orchid", "Phoenix", "Jade", "Crystal", "Maple", "Falcon", "Nova", "Apex", "Delta", "Sun", "Star", "Ocean"]
NAME_B = ["Precision", "Electronics", "Components", "Materials", "Industries", "Technology", "Metals", "Polymers", "Circuits", "Power", "Optics", "Systems"]
CERTS = ["ISO 9001", "ISO 14001", "ISO 45001", "IATF 16949", "RBA VAP Silver", "RBA VAP Gold", "RMAP conformant smelter", "IECQ QC 080000"]
PRODUCTS = [
    ("PSU-1200", "1200 W server power supply"), ("PSU-800", "800 W industrial power supply"), ("BBU-48", "48 V battery backup unit"),
    ("ADPT-65", "65 W USB-C adapter"), ("ADPT-140", "140 W GaN laptop adapter"), ("LED-HB", "High-bay LED luminaire"),
    ("LED-ST", "Smart street light"), ("CHG-EV7", "7 kW EV charger"), ("CHG-EV22", "22 kW EV charger"), ("INV-5K", "5 kW solar inverter"),
    ("UPS-3K", "3 kVA online UPS"), ("UPS-10K", "10 kVA online UPS"), ("SNS-IoT", "Industrial IoT sensor node"), ("GW-5G", "5G industrial gateway"),
    ("CAM-AI", "Edge AI camera module"), ("KBD-M", "Mechanical keyboard"), ("MSE-W", "Wireless mouse"), ("DSP-27", "27-inch monitor power board"),
    ("FAN-CTRL", "Smart fan controller"), ("RLY-PWR", "Power relay module"),
]
MARKETS = ["EU", "US", "TW", "JP", "CN", "IN"]


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def gen_suppliers(n: int, rng: random.Random) -> list[Supplier]:
    mats = list(MATERIAL_TOPIC)
    out = []
    for i in range(1, n + 1):
        country = rng.choice(COUNTRIES)
        name = f"{rng.choice(NAME_A)} {rng.choice(NAME_B)} ({country[:2].upper()}{i:03d})"
        k = rng.randint(1, 4)
        materials = rng.sample(mats, k)
        certs = rng.sample(CERTS, rng.randint(1, 4))
        flags = []
        if rng.random() < 0.06:
            flags.append("late RBA corrective action")
        if rng.random() < 0.04:
            flags.append("single-source for a critical part")
        if country in ("China",) and rng.random() < 0.15:
            flags.append("entity-list screening pending")
        out.append(Supplier(supplier_id=f"S{i:04d}", name=name, country=country, tier=rng.choices([1, 2, 3], [0.5, 0.35, 0.15])[0],
                            certifications=certs, materials=materials, rba_audit_score=rng.randint(90, 200), risk_flags=flags,
                            contact_email=f"compliance@{name.split()[0].lower()}{i:03d}.example.com"))
    return out


def gen_parts(suppliers: list[Supplier], rng: random.Random) -> list[Part]:
    parts = []
    pid = 1
    for s in suppliers:
        for m in s.materials:
            for _ in range(rng.randint(1, 2)):
                parts.append(Part(part_id=f"P{pid:05d}", description=f"{m} - {rng.choice(['grade A', 'grade B', 'automotive', 'industrial', 'consumer'])}",
                                  material=m, supplier_id=s.supplier_id, unit_cost_usd=round(rng.uniform(0.05, 120), 2)))
                pid += 1
    return parts


def gen_products(parts: list[Part], rng: random.Random) -> list[Product]:
    products = []
    for code, name in PRODUCTS:
        n = rng.randint(12, 30)
        chosen = rng.sample(parts, n)
        products.append(Product(product_id=code, name=name, markets=rng.sample(MARKETS, rng.randint(2, 5)), bom=[(p.part_id, rng.randint(1, 8)) for p in chosen]))
    return products


# ---------------- documents ----------------

POLICY_TOPICS: list[tuple[str, Topic, str, list[tuple[str, str]]]] = [
    ("POL-ENV-001", Topic.PFAS, "Per- and polyfluoroalkyl substances (PFAS) restriction policy", [
        ("1 Purpose", "Defines how the company controls PFAS in products, packaging and manufacturing consumables ahead of the EU universal PFAS restriction under REACH and US state laws (Maine, Minnesota)."),
        ("2 Scope", "Applies to all parts containing PTFE coatings, fluoropolymer gaskets or PFAS surfactants, and to all suppliers of such parts."),
        ("3 Requirements", "Suppliers must declare PFAS content above 25 ppb per substance and 250 ppb total. Engineering must maintain an approved substitute list. New designs must not introduce PFAS without a documented exemption."),
        ("4 Response to regulatory change", "Within 10 working days of a published restriction or consultation, Compliance issues an impact assessment listing affected parts, products and suppliers, and Procurement requests updated declarations from each affected supplier within 30 days."),
    ]),
    ("POL-ENV-002", Topic.ROHS, "RoHS compliance policy", [
        ("1 Scope", "All electrical and electronic equipment placed on EU, UK, China, Taiwan and other RoHS-type markets."),
        ("2 Restricted substances", "Lead, mercury, cadmium, hexavalent chromium, PBB, PBDE, DEHP, BBP, DBP, DIBP above the 0.1 percent (cadmium 0.01 percent) thresholds in homogeneous materials, unless an exemption applies."),
        ("3 Exemptions", "Exemption 7(a) lead in high-melting-temperature solder is tracked with its expiry date; the exemption register is reviewed quarterly."),
        ("4 Supplier obligations", "Material declarations in IPC-1752A format for every part; re-declaration when an exemption expires or a substance is added."),
    ]),
    ("POL-ENV-003", Topic.REACH_SVHC, "REACH SVHC and Candidate List monitoring", [
        ("1 Monitoring", "Compliance monitors ECHA Candidate List updates (normally January and June). Each new SVHC triggers a parts screening within 15 working days."),
        ("2 Communication duty", "Articles containing an SVHC above 0.1 percent w/w require Article 33 communication to customers and SCIP database notification."),
        ("3 Phthalates and flame retardants", "DEHP, BBP, DBP, DIBP and brominated flame retardants are tracked at part level with supplier declarations."),
    ]),
    ("POL-SRC-004", Topic.CONFLICT_MINERALS, "Responsible minerals sourcing policy (3TG and cobalt)", [
        ("1 Commitment", "Tin, tantalum, tungsten, gold and cobalt are sourced only from smelters conformant with the Responsible Minerals Assurance Process (RMAP)."),
        ("2 Supplier reporting", "Suppliers of 3TG- or cobalt-bearing parts submit a CMRT and EMRT annually and within 30 days of a smelter list change."),
        ("3 Escalation", "A non-conformant smelter in the chain triggers a 60-day corrective plan; failure leads to supplier phase-out."),
    ]),
    ("POL-SR-005", Topic.RBA_LABOUR, "Responsible Business Alliance (RBA) code adoption", [
        ("1 Adoption", "The RBA Code of Conduct applies to the company and to all tier-1 suppliers; tier-2 suppliers are covered through flow-down clauses."),
        ("2 Audits", "Tier-1 suppliers with RBA audit scores below 160 receive a corrective action plan with 90-day closure."),
        ("3 Forced labour", "Zero tolerance for forced labour indicators per ILO; recruitment fees are repaid to workers; supplier facilities in high-risk regions are audited annually."),
    ]),
    ("POL-TRD-006", Topic.EXPORT_CONTROL, "Export control and sanctions screening", [
        ("1 Screening", "All suppliers and customers are screened against the US Entity List, EU and UN sanctions lists, and Taiwan SHTC lists at onboarding and weekly thereafter."),
        ("2 Controlled items", "High-performance FPGAs, GaN RF amplifiers and encryption modules require ECCN classification and licence determination before shipment."),
        ("3 Response", "A new listing affecting a supplier freezes open purchase orders pending Legal review within 2 working days."),
    ]),
    ("POL-ESG-007", Topic.CSRD_ESG, "Sustainability reporting (CSRD / ESRS) data policy", [
        ("1 Scope", "Scope 1, 2 and 3 emissions, supplier ESG data and double materiality assessment for EU reporting."),
        ("2 Supplier data", "Tier-1 suppliers provide annual emissions and ESG questionnaires; data is assured before publication."),
    ]),
    ("POL-ENV-008", Topic.BATTERY, "EU Battery Regulation compliance", [
        ("1 Scope", "Lithium-ion and NiMH cells and packs: carbon footprint declaration, due diligence, battery passport readiness, recycled content thresholds."),
        ("2 Supplier obligations", "Cell suppliers provide carbon footprint data per EU methodology and due-diligence reports."),
    ]),
    ("POL-ENV-009", Topic.PACKAGING, "Packaging and packaging waste policy", [
        ("1 Requirements", "EPS foam and PVC packaging are being phased out by 2027; recycled content targets per EU PPWR."),
    ]),
    ("POL-SEC-010", Topic.CYBERSECURITY, "Product cybersecurity (EU Cyber Resilience Act) policy", [
        ("1 Scope", "Connected products (gateways, sensors, cameras, chargers) require SBOM, vulnerability handling and secure update processes."),
    ]),
    ("POL-PRV-011", Topic.DATA_PROTECTION, "Personal data protection policy", [
        ("1 Scope", "GDPR and Taiwan PDPA apply to employee, customer and supplier-contact data. Right-to-erasure requests are completed within 30 days, including removal from AI knowledge bases, with verification evidence retained."),
    ]),
    ("POL-GOV-012", Topic.OTHER, "Regulatory change management procedure", [
        ("1 Intake", "Regulatory changes are logged within 1 working day of publication with source, jurisdiction and topic."),
        ("2 Impact assessment", "An impact brief identifies affected products, parts, suppliers and contract clauses, each with evidence, and proposes actions with owners and deadlines."),
        ("3 Approval", "Briefs are approved by the Compliance Manager before tasks are issued. Automated drafts are never actioned without approval."),
        ("4 Urgency", "High urgency: enforcement within 6 months or a sanctions listing. Medium: within 24 months. Low: consultation stage."),
    ]),
]

CONTRACT_CLAUSES = [
    ("1 Compliance with laws", "Supplier shall comply with all applicable product environmental laws including RoHS, REACH, PFAS restrictions and the EU Battery Regulation, and shall notify Buyer within 15 days of any change in the regulatory status of supplied materials."),
    ("2 Material declarations", "Supplier shall provide full material declarations (IPC-1752A) and substance declarations on request and at least annually; declarations shall be updated within 30 days of any new restriction."),
    ("3 Responsible sourcing", "Supplier shall source 3TG and cobalt from RMAP-conformant smelters and submit CMRT/EMRT annually."),
    ("4 Labour and ethics", "Supplier shall adhere to the RBA Code of Conduct and permit audits with 10 days notice."),
    ("5 Export control", "Supplier represents that it is not listed on any sanctions or entity list and shall notify Buyer immediately if that changes; Buyer may suspend orders without liability."),
    ("6 Change notification", "Supplier shall notify Buyer 90 days before any change of material, process, sub-supplier or manufacturing location (PCN)."),
    ("7 Remedies", "Non-compliance with clauses 1 to 6 entitles Buyer to reject goods, require replacement at Supplier's cost, and terminate for cause after a 30-day cure period."),
    ("8 Term", "This agreement runs for 36 months and renews annually unless terminated with 90 days notice."),
]


def gen_documents(suppliers: list[Supplier], rng: random.Random) -> list[dict]:
    docs = []
    for doc_id, topic, title, sections in POLICY_TOPICS:
        docs.append({"doc_id": doc_id, "doc_type": "policy", "title": title, "topics": [topic.value], "jurisdiction": "GLOBAL", "supplier_id": None,
                     "effective_date": f"202{rng.randint(3, 6)}-0{rng.randint(1, 9)}-01", "sections": sections})
    # More policies/SOPs: per-topic operational SOPs so retrieval has near-duplicates to rank
    sop_id = 101
    for _, topic, _title, _ in POLICY_TOPICS[:10]:
        docs.append({"doc_id": f"SOP-{sop_id}", "doc_type": "sop", "title": f"SOP: handling a regulatory change on {topic.value.replace('_', ' ')}", "topics": [topic.value], "jurisdiction": "GLOBAL", "supplier_id": None,
                     "effective_date": "2025-01-15", "sections": [
                         ("1 Trigger", f"A published or proposed change relating to {topic.value.replace('_', ' ')} is logged as an event."),
                         ("2 Steps", "Identify affected parts by material; list suppliers of those parts; check contract clauses 1, 2 and 6 for notification duties; draft actions with owners."),
                         ("3 Records", "Store the brief, evidence and approvals in the compliance register."),
                     ]})
        sop_id += 1
    # Contracts for 30 suppliers (tier 1 preferred)
    tier1 = [s for s in suppliers if s.tier == 1]
    for s in rng.sample(tier1, min(30, len(tier1))):
        cid = f"CTR-{s.supplier_id}"
        docs.append({"doc_id": cid, "doc_type": "contract", "title": f"Supply agreement with {s.name}", "topics": [MATERIAL_TOPIC[m].value for m in s.materials], "jurisdiction": "GLOBAL",
                     "supplier_id": s.supplier_id, "effective_date": f"202{rng.randint(2, 5)}-{rng.randint(1, 12):02d}-01",
                     "sections": [(h, t.replace("Supplier", s.name.split(" (")[0] + " (Supplier)", 1) if h.startswith("1") else t) for h, t in CONTRACT_CLAUSES]})
    return docs


def write_all(out_dir: Path, seed: int = 7) -> dict:
    rng = _rng(seed)
    suppliers = gen_suppliers(200, rng)
    parts = gen_parts(suppliers, rng)
    products = gen_products(parts, rng)
    docs = gen_documents(suppliers, rng)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "suppliers.json").write_text(json.dumps([s.model_dump() for s in suppliers], indent=1), encoding="utf-8")
    (out_dir / "parts.json").write_text(json.dumps([p.model_dump() for p in parts], indent=1), encoding="utf-8")
    (out_dir / "products.json").write_text(json.dumps([p.model_dump() for p in products], indent=1), encoding="utf-8")
    (out_dir / "documents.json").write_text(json.dumps(docs, indent=1), encoding="utf-8")
    docs_dir = out_dir / "docs"
    docs_dir.mkdir(exist_ok=True)
    for d in docs:
        md = [f"# {d['doc_id']}: {d['title']}", ""]
        for h, t in d["sections"]:
            md += [f"## {h}", "", t, ""]
        (docs_dir / f"{d['doc_id']}.md").write_text("\n".join(md), encoding="utf-8")
    meta = {"seed": seed, "suppliers": len(suppliers), "parts": len(parts), "products": len(products), "documents": len(docs),
            "by_type": {t: sum(1 for d in docs if d["doc_type"] == t) for t in ("policy", "sop", "contract")}}
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=settings.generated_dir)
    a = ap.parse_args()
    print(json.dumps(write_all(a.out, a.seed)))
