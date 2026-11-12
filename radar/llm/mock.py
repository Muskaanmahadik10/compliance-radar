"""MockChatModel: a LangChain-compatible chat model that answers from rules so the whole pipeline (graph, tools, schemas,
evals, CI) runs with no key. It implements the structured-output path by filling the requested pydantic schema from
the prompt contents. It is deliberately simple and deterministic.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel


class MockChatModel(BaseChatModel):
    model_name: str = "mock"
    _schema: type[BaseModel] | None = None

    @property
    def _llm_type(self) -> str:
        return "mock"

    def with_structured_output(self, schema: Any, **kwargs):  # type: ignore[override]
        m = MockChatModel(model_name=self.model_name)
        m._schema = schema
        return _StructuredMock(m, schema)

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager: CallbackManagerForLLMRun | None = None, **kwargs) -> ChatResult:
        text = "\n".join(str(m.content) for m in messages)
        answer = self._answer(text)
        msg = AIMessage(content=answer, usage_metadata={"input_tokens": len(text) // 4, "output_tokens": len(answer) // 4, "total_tokens": len(text) // 4 + len(answer) // 4})
        return ChatResult(generations=[ChatGeneration(message=msg)])

    # ---- rule-based behaviours keyed by TASK markers in the prompt ----
    def _answer(self, text: str) -> str:
        if "TASK: decompose" in text:
            return json.dumps({"sub_questions": [
                "Which policy sections govern this topic and what deadlines do they set?",
                "Which parts and suppliers use the affected materials?",
                "Which contract clauses require supplier notification or declarations?",
            ]})
        if "TASK: chat" in text:
            cites = re.findall(r"\[(\S+ §[^\]]+)\]", text)
            return f"Based on the retrieved documents: {text[-400:].strip()[:200]}... (see {', '.join(dict.fromkeys(cites[:3])) or 'no sources'})"
        if "Classify this regulatory" in text:
            return "other"
        return "ok"


class _StructuredMock:
    """Returns an instance of `schema` built from whatever evidence appears in the prompt."""

    def __init__(self, model: MockChatModel, schema: type[BaseModel]):
        self.model, self.schema = model, schema

    def invoke(self, messages, config=None, **kwargs):
        text = "\n".join(str(getattr(m, "content", m)) for m in (messages if isinstance(messages, list) else [messages]))
        name = self.schema.__name__
        if name == "ImpactBrief":
            return self._brief(text)
        if name == "CriticVerdict":
            return self.schema(approved=True, missing_citations=[], notes="mock critic: all claims carry citations")
        if name == "SubQuestions":
            return self.schema(sub_questions=["policy sections and deadlines", "affected parts and suppliers", "contract notification clauses"])
        return self.schema.model_construct()

    def _brief(self, text: str):
        from radar.data.schemas import AffectedItem, Citation, ImpactBrief, Jurisdiction, RequiredAction, Topic

        ev = re.search(r'"event_id":\s*"([^"]+)"', text)
        title = re.search(r'"title":\s*"([^"]+)"', text)
        topic = re.search(r'"topic":\s*"([^"]+)"', text)
        juris = re.search(r'"jurisdiction":\s*"([^"]+)"', text)
        cites = [Citation(doc_id=d, section=s.strip(), quote="") for d, s in dict.fromkeys(re.findall(r"\[([A-Z]+-[A-Z0-9-]+) §([^\]]+)\]", text))][:8]
        suppliers = [(sid, nm) for sid, nm in dict.fromkeys(re.findall(r'"supplier_id":\s*"(S\d{4})",\s*"name":\s*"([^"]+)"', text))][:6]
        parts = list(dict.fromkeys(re.findall(r'"part_id":\s*"(P\d{5})"', text)))[:6]
        affected = [AffectedItem(kind="supplier", id=sid, name=nm, why="supplies parts containing an affected material", citations=cites[:1]) for sid, nm in suppliers]
        affected += [AffectedItem(kind="part", id=p, name=p, why="contains an affected material", citations=cites[:1]) for p in parts]
        t = topic.group(1) if topic and topic.group(1) in {x.value for x in Topic} else "other"
        j = juris.group(1) if juris and juris.group(1) in {x.value for x in Jurisdiction} else "GLOBAL"
        return ImpactBrief(
            event_id=ev.group(1) if ev else "unknown", title=title.group(1)[:200] if title else "Impact brief", topic=Topic(t), jurisdiction=Jurisdiction(j),
            summary=f"Regulatory change on {t.replace('_', ' ')} affects {len(suppliers)} suppliers and {len(parts)} parts per the cited policy and contract clauses. Mock summary.",
            affected=affected or [AffectedItem(kind="policy", id=cites[0].doc_id if cites else "POL-GOV-012", name="policy", why="governs response", citations=cites[:1])],
            actions=[RequiredAction(action="Request updated supplier declarations", owner_role="Procurement", deadline="30 days", citations=cites[:1]),
                     RequiredAction(action="Screen parts list and update substitute register", owner_role="Compliance", deadline="10 working days", citations=cites[1:2] or cites[:1])],
            confidence=0.7 if cites else 0.3, open_questions=[] if cites else ["No supporting documents were retrieved."], citations=cites or [Citation(doc_id="POL-GOV-012", section="2 Impact assessment", quote="")],
            requires_approval=True,
        )
