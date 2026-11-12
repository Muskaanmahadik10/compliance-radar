"""Streamlit front end. Talks to the same service functions as the API (in-process), so `streamlit run ui/app.py`
works without starting anything else in local mode.

Tabs: Live events -> Briefs (review and approve) -> Ask the knowledge base -> Documents (verified erasure) -> Metrics
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st
from sqlalchemy import select

from radar import chat
from radar.agent import service
from radar.config import ARTIFACTS_DIR, DATA_DIR, settings
from radar.db import create_all
from radar.db.schema import documents, events
from radar.forget import forget_document, verify_certificate
from radar.infra import get_db_engine, health
from radar.ingest import feeds, worker
from radar.llm import get_router

st.set_page_config(page_title="ComplianceRadar", layout="wide")


@st.cache_resource
def bootstrap() -> dict:
    """First run in local mode: build the demo company and index it so the app opens on real content."""
    create_all()
    from radar.retrieval import DocumentStore

    built = False
    if not (DATA_DIR / "generated" / "suppliers.json").exists() or DocumentStore().count() == 0:
        from radar.data.build import build_all
        from radar.retrieval.ingest import ingest

        build_all()
        ingest(recreate=True)
        built = True
    return {"built": built}


bootstrap()
URGENCY = {"high": "🔴 high", "medium": "🟠 medium", "low": "🟢 low"}


def q(sql):
    with get_db_engine().connect() as c:
        return [dict(r) for r in c.execute(sql).mappings()]


st.title("ComplianceRadar")
st.caption(f"Real-time compliance and supplier-risk copilot · mode **{settings.mode}** · models **{get_router().names['cheap']}** (steps) / **{get_router().names['strong']}** (drafting) · every brief needs human approval")

tab_events, tab_briefs, tab_chat, tab_docs, tab_metrics = st.tabs(["Live events", "Briefs", "Ask the knowledge base", "Documents", "Metrics"])

# ---------------- events ----------------
with tab_events:
    c1, c2, c3 = st.columns([1, 1, 4])
    if c1.button("Poll feeds (replay)", help="Publishes the demo feed items to the stream and processes them"):
        n = feeds.poll_once(use_network=False, replay=True)
        st.toast(f"published {n} · {worker.process_available()}")
    if c2.button("Poll live feeds", help="EUR-Lex, ECHA and Federal Register RSS (needs internet)"):
        n = feeds.poll_once(use_network=True, replay=False)
        st.toast(f"published {n} · {worker.process_available()}")
    evs = q(select(events).order_by(events.c.published.desc()).limit(200))
    if not evs:
        st.info("No events yet. Press **Poll feeds (replay)** to load the demo stream: 13 items, of which 1 is a near-duplicate and 1 is irrelevant news.")
    else:
        relevant = [e for e in evs if e["status"] != "ignored"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Events stored", len(evs))
        m2.metric("Relevant", len(relevant))
        m3.metric("High urgency", sum(e["urgency"] == "high" for e in relevant))
        m4.metric("Awaiting brief", sum(e["status"] == "new" for e in relevant))
        for e in relevant:
            with st.container(border=True):
                a, b = st.columns([5, 1])
                a.markdown(f"**{e['title']}**  \n{URGENCY[e['urgency']]} · `{e['topic']}` · {e['jurisdiction']} · {e['source']} · {e['published'][:10]} · status `{e['status']}`")
                a.caption(e["summary"][:300])
                if e["status"] == "new" and b.button("Draft brief", key=f"b{e['event_id']}", type="primary"):
                    with st.spinner("Researching, mapping impact, drafting, verifying..."):
                        res = service.start_brief(e["event_id"])
                    st.session_state["open_brief"] = res["brief_id"]
                    st.toast(f"Brief {res['brief_id']} is waiting for review (see Briefs tab)")
                    st.rerun()

# ---------------- briefs ----------------
with tab_briefs:
    bl = service.list_briefs()
    if not bl:
        st.info("No briefs yet. Draft one from the Live events tab.")
    else:
        ids = [b["brief_id"] for b in bl]
        default = ids.index(st.session_state["open_brief"]) if st.session_state.get("open_brief") in ids else 0
        pick = st.selectbox("Brief", ids, index=default, format_func=lambda i: f"{i} · {next(b['status'] for b in bl if b['brief_id'] == i)}")
        b = service.get_brief(pick)
        br = b["brief"]
        st.subheader(br["title"])
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Status", b["status"])
        k2.metric("Suppliers affected", br["affected_counts"].get("suppliers", 0))
        k3.metric("Parts / products", f"{br['affected_counts'].get('parts', 0)} / {br['affected_counts'].get('products', 0)}")
        k4.metric("Confidence", f"{br['confidence']:.0%}")
        k5.metric("Cost / time", f"${b['cost_usd']:.4f} / {b['latency_s']:.0f}s")
        st.write(br["summary"])
        st.caption(f"Drafted by {br.get('_drafted_by', 'n/a')} · critic {br.get('_verdict', {}).get('critic_model') or 'n/a'} · {br.get('_n_evidence', 0)} evidence passages")
        if br.get("_auto_findings"):
            st.warning("Verifier: " + " · ".join(br["_auto_findings"]))
        v = br.get("_verdict", {})
        if v.get("critic_ran") is False:
            st.info(f"Critic: not run. {v.get('notes', '')}")
        else:
            (st.success if v.get("approved") else st.error)(f"Critic: {'approved' if v.get('approved') else 'not approved'}. {v.get('notes', '')}")
            for mc in v.get("missing_citations", []):
                st.markdown(f"- {mc}")
        left, right = st.columns(2)
        with left:
            st.markdown("**Required actions**")
            for i, a in enumerate(br["actions"], 1):
                cites = ", ".join(f"{c['doc_id']} §{c['section']}" for c in a["citations"]) or "no citation"
                st.markdown(f"{i}. **{a['action']}**")
                st.caption(f"Owner: {a['owner_role']} · Deadline: {a['deadline']} · Source: {cites}")
            if br.get("open_questions"):
                st.markdown("**Open questions**")
                for oq in br["open_questions"]:
                    st.markdown(f"- {oq}")
        with right:
            st.markdown("**Most affected**")
            for a in br["affected"]:
                st.markdown(f"- `{a['kind']}` **{a['id']}** {a['name']}: {a['why']}")
        with st.expander(f"Evidence cited ({len(br['citations'])})"):
            for c in br["citations"]:
                st.markdown(f"**{c['doc_id']} §{c['section']}**  \n> {c.get('quote', '')}")
        if b["status"] == "pending_review":
            who = st.text_input("Reviewer", "compliance.manager")
            a, r = st.columns(2)
            if a.button("Approve and issue tasks", type="primary"):
                service.decide(pick, True, who)
                st.rerun()
            if r.button("Reject"):
                service.decide(pick, False, who)
                st.rerun()
        else:
            st.markdown(f"Decided by **{b['decided_by']}** at {b['decided_at']}")
            if b["tasks"]:
                st.markdown("**Tasks issued**")
                st.dataframe(pd.DataFrame(b["tasks"])[["task_id", "action", "owner_role", "deadline", "status"]], hide_index=True, use_container_width=True)

# ---------------- chat ----------------
with tab_chat:
    st.caption("Answers come only from the indexed policies, SOPs and contracts, with citations. Try: *Do we have to tell customers when a product contains a hazardous chemical?*")
    hist = st.session_state.setdefault("chat", [])
    for m in hist:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    if prompt := st.chat_input("Ask about policies, contracts or supplier obligations"):
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            box, text, final = st.empty(), "", {}
            sources = []
            for ev in chat.stream_answer(prompt, hist):
                if ev["type"] == "sources":
                    sources = ev["hits"]
                elif ev["type"] == "token":
                    text += ev["text"]
                    box.markdown(text + "▌")
                else:
                    final = ev
            text = final.get("answer", text)  # normalised punctuation, so citations render uniformly
            box.markdown(text)
            if final.get("citations"):
                st.caption("Verified citations: " + ", ".join(f"{c['doc_id']} §{c['section']}" for c in final["citations"]))
            if final.get("invalid_citations"):
                st.warning("Citations not found in the retrieved passages: " + ", ".join(final["invalid_citations"]))
            with st.expander(f"Sources ({len(sources)}) · cost ${final.get('cost', {}).get('cost_usd', 0):.5f} · PII redacted: {final.get('pii_redacted', 0)}"):
                for h in sources:
                    st.markdown(f"**{h['citation']}** ({h['doc_type']}, score {h['score']})  \n{h['text'][:300]}")
        hist += [{"role": "user", "content": prompt}, {"role": "assistant", "content": text}]

# ---------------- documents ----------------
with tab_docs:
    docs = q(select(documents).order_by(documents.c.doc_type, documents.c.doc_id))
    st.dataframe(pd.DataFrame(docs)[["doc_id", "doc_type", "title", "status", "n_chunks", "effective_date"]], hide_index=True, use_container_width=True, height=280)
    st.markdown("#### Erase a document (right to be forgotten)")
    st.caption("Deletes the document from the vector store, the database, disk and caches, then verifies with probe questions that it can no longer be retrieved or cited, and issues a signed certificate.")
    active = [d["doc_id"] for d in docs if d["status"] == "active"]
    d1, d2, d3 = st.columns([2, 2, 1])
    target = d1.selectbox("Document", active)
    who = d2.text_input("Requested by", "dpo@company.example")
    if d3.button("Erase and verify", type="primary") and target:
        with st.spinner("Deleting and verifying..."):
            cert = forget_document(target, who)
        (st.success if cert["verified"] else st.error)(f"Erasure of {target}: {'verified' if cert['verified'] else 'NOT verified'} · signature valid: {verify_certificate(cert)}")
        st.json(cert)
    certs = sorted((ARTIFACTS_DIR / "certificates").glob("*.json")) if (ARTIFACTS_DIR / "certificates").exists() else []
    if certs:
        with st.expander(f"Certificates on file ({len(certs)})"):
            for p in certs:
                c = json.loads(p.read_text(encoding="utf-8"))
                st.markdown(f"- `{c['doc_id']}` · {c['completed_at'][:19]} · verified **{c['verified']}** · signature valid **{verify_certificate(c)}**")

# ---------------- metrics ----------------
with tab_metrics:
    st.json(health())
    for name, title in (("retrieval_eval", "Retrieval"), ("brief_eval_env", "Briefs (real models)"), ("brief_eval_mock", "Briefs (mock, pipeline check)")):
        p = ARTIFACTS_DIR / f"{name}.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        st.markdown(f"#### {title}")
        if name == "retrieval_eval":
            st.caption("Paraphrased questions (the set that separates methods)")
            st.dataframe(pd.DataFrame(data["paraphrased"]), hide_index=True)
            st.caption("Direct questions")
            st.dataframe(pd.DataFrame(data["direct"]), hide_index=True)
        else:
            st.json(data["summary"])
            st.dataframe(pd.DataFrame(data["rows"]), hide_index=True)
