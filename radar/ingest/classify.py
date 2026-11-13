"""Event classification: topic, jurisdiction, urgency, language.

Three tiers, cheapest first:
  1. rule-based keyword classifier (always available, used by tests and as a fallback)
  2. fine-tuned LoRA encoder (optional; RADAR_CLASSIFIER_MODEL, see radar/ingest/lora_classifier.py)
  3. LLM zero-shot (cheap model) when the first two are not confident
Returns a label plus a confidence and which tier decided, so the evaluation can compare tiers on cost and F1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langdetect import DetectorFactory, LangDetectException, detect

from radar.data.schemas import Jurisdiction, Topic

DetectorFactory.seed = 0

TOPIC_KEYWORDS: dict[Topic, list[str]] = {
    Topic.PFAS: ["pfas", "全氟", "多氟烷基", "per- and polyfluoroalkyl", "perfluoro", "ptfe", "fluoropolymer", "forever chemicals"],
    Topic.ROHS: ["rohs", "限用物質", "restriction of hazardous substances", "lead in solder", "cadmium", "hexavalent chromium", "exemption 7"],
    Topic.REACH_SVHC: ["svhc", "candidate list", "reach", "authorisation list", "annex xiv", "annex xvii", "phthalate", "dehp", "bisphenol"],
    Topic.CONFLICT_MINERALS: ["conflict mineral", "衝突礦產", "3tg", "tantalum", "tungsten", "cobalt", "smelter", "rmap", "cmrt", "due diligence regulation"],
    Topic.RBA_LABOUR: ["forced labour", "forced labor", "uflpa", "rba", "responsible business alliance", "modern slavery", "labour rights"],
    Topic.EXPORT_CONTROL: ["entity list", "export control", "bis ", "bureau of industry and security", "eccn", "sanction", "ofac", "dual-use", "出口管制", "戰略性高科技貨品", "輸出許可", "實體清單"],
    Topic.CSRD_ESG: ["csrd", "esrs", "sustainability reporting", "scope 3", "double materiality", "taxonomy"],
    Topic.BATTERY: ["battery regulation", "battery passport", "lithium-ion", "batteries regulation", "2023/1542"],
    Topic.PACKAGING: ["packaging", "ppwr", "packaging waste", "recycled content"],
    Topic.CYBERSECURITY: ["cyber resilience act", "cybersecurity", "vulnerability", "sbom", "nis2"],
    Topic.DATA_PROTECTION: ["gdpr", "personal data", "data protection", "pdpa", "privacy"],
    Topic.SUPPLIER_INCIDENT: ["factory fire", "fire at", "halts production", "explosion", "bankruptcy", "insolvency", "strike", "flood", "earthquake", "recall", "plant shutdown", "工廠火災", "停產"],
}
JURIS_KEYWORDS: dict[Jurisdiction, list[str]] = {
    Jurisdiction.EU: ["eu ", "european", "commission", "echa", "eur-lex", "official journal", "regulation (eu)", "directive"],
    Jurisdiction.US: ["epa", "federal register", "bis", "u.s.", "united states", "california", "maine", "minnesota", "tsca"],
    Jurisdiction.TW: ["taiwan", "moea", "bsmi", "經濟部", "標準檢驗局", "台灣", "臺灣"],
    Jurisdiction.CN: ["china", "mee", "gb/t", "中國", "中国"],
}
URGENT = ["enters into force", "effective immediately", "entity list", "ban", "prohibited", "recall", "fire", "within 30 days", "final rule"]
LOW = ["consultation", "proposal", "draft", "call for evidence", "roadmap", "guidance", "faq"]


@dataclass
class Classification:
    topic: Topic
    jurisdiction: Jurisdiction
    urgency: str
    language: str
    confidence: float
    tier: str  # rules | lora | llm


_HAN = re.compile(r"[一-鿿]")
# An operational incident outranks the material it happens to mention ("fire at a tantalum capacitor plant" is an incident).
TOPIC_WEIGHT = {Topic.SUPPLIER_INCIDENT: 3}


def detect_language(text: str) -> str:
    # langdetect is unreliable on short CJK strings; a Han-character ratio is a sturdier first check.
    stripped = re.sub(r"\s", "", text[:500])
    if stripped and len(_HAN.findall(stripped)) / len(stripped) > 0.2:
        return "zh"
    try:
        return detect(text[:500]) if text.strip() else "en"
    except LangDetectException:
        return "en"


def classify_rules(title: str, summary: str = "") -> Classification:
    text = f"{title} {summary}".lower()
    scores = {t: TOPIC_WEIGHT.get(t, 1) * sum(len(re.findall(re.escape(k), text)) for k in kws) for t, kws in TOPIC_KEYWORDS.items()}
    topic = max(scores, key=scores.get)
    total = sum(scores.values())
    if scores[topic] == 0:
        topic, conf = Topic.OTHER, 0.3
    else:
        conf = min(0.95, 0.5 + 0.45 * scores[topic] / max(total, 1))
    jscores = {j: sum(text.count(k) for k in kws) for j, kws in JURIS_KEYWORDS.items()}
    juris = max(jscores, key=jscores.get) if max(jscores.values()) > 0 else Jurisdiction.GLOBAL
    urgency = "high" if any(k in text for k in URGENT) else "low" if any(k in text for k in LOW) else "medium"
    return Classification(topic, juris, urgency, detect_language(title + " " + summary), conf, "rules")


def llm_callback(router=None, callbacks: list | None = None):
    """The LLM tier as a callable(prompt) -> str, built from the cheap model of the router.

    This is the one place the third tier is constructed: the stream worker, the API, the UI and evaluation/classifier.py
    all get it from here. Returns None when the router is the offline mock, so a deterministic stand-in is never
    presented as a live LLM classification (offline runs use rules, plus LoRA if configured).
    """
    if router is None:
        from radar.llm import get_router

        router = get_router()
    model = router.cheap
    if getattr(model, "_llm_type", "") == "mock":
        return None
    cfg = {"callbacks": list(callbacks or [])}

    def _call(prompt: str) -> str:
        return str(model.invoke(prompt, config=cfg).content)

    _call.model_name = getattr(model, "model_name", None) or getattr(model, "model", "?")  # type: ignore[attr-defined]
    return _call


def active_tiers() -> list[str]:
    """Which tiers live ingestion would use right now (shown by /health)."""
    from radar.ingest import lora_classifier

    tiers = ["rules"]
    if lora_classifier.get() is not None:
        tiers.append("lora")
    try:
        if llm_callback() is not None:
            tiers.append("llm")
    except Exception:
        pass
    return tiers


def classify(title: str, summary: str = "", llm=None, lora=None, lora_min_confidence: float | None = None) -> Classification:
    """Tiered classification, cheapest first:

      1. rules      - free; trusted when a keyword matched clearly (confidence >= 0.6)
      2. lora       - the fine-tuned encoder, if configured; trusted above `lora_min_confidence`
      3. llm        - `llm` is an optional callable(prompt) -> str for whatever is still uncertain
    Falls back to the best earlier guess if a later tier is unavailable or fails.
    """
    c = classify_rules(title, summary)
    if c.confidence >= 0.6:
        return c
    best = c
    if lora is None:
        from radar.ingest import lora_classifier

        lora = lora_classifier.get()
    if lora is not None:
        from radar.config import settings

        thr = settings.classifier_lora_min_confidence if lora_min_confidence is None else lora_min_confidence
        try:
            label, conf = lora.predict([title])[0]
            guess = Classification(Topic(label), c.jurisdiction, c.urgency, c.language, conf, "lora")
            if conf >= thr:
                return guess
            best = guess
        except Exception:
            pass
    if llm is not None:
        try:
            topics = ", ".join(t.value for t in Topic)
            prompt = "\n".join([f"Classify this regulatory/news item into exactly one topic from [{topics}]. Reply with the topic id only.",
                                f"Title: {title}", f"Summary: {summary[:800]}"])
            ans = llm(prompt)
            t = ans.strip().lower().split()[0].strip(".,'\"")
            if t in {x.value for x in Topic}:
                return Classification(Topic(t), c.jurisdiction, c.urgency, c.language, 0.7, "llm")
        except Exception:
            pass
    return best
