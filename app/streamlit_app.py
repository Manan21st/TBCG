"""TBCG Streamlit dashboard.

Submit a business problem, watch the multi-agent workflow stream node-by-node,
approve/reject the recommendation (human-in-the-loop), and download the final
consulting report as a PDF.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import streamlit as st

# Make ``src`` importable.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from langgraph.types import Command  # noqa: E402

from tbcg.config import get_settings  # noqa: E402
from tbcg.graph.build import build_graph  # noqa: E402
from tbcg.state import initial_state  # noqa: E402
from tbcg.tools.rag import chroma_available  # noqa: E402
from tbcg.tools.report import export_pdf_bytes  # noqa: E402

NODE_LABELS = {
    "engagement": "🧭 Engagement Manager",
    "research": "🔎 Research Agent",
    "finance": "💰 Financial Analyst",
    "strategy": "🎯 Strategy Consultant",
    "risk": "⚠️ Risk Analyst",
    "critic": "🧐 Critic",
    "revise": "🔄 Revision",
    "human_review": "🙋 Human Review",
    "report": "📄 Report",
}

st.set_page_config(page_title="TBCG — TechBro Consulting Group", page_icon="🧠", layout="wide")


# --------------------------------------------------------------------------- #
# Session helpers
# --------------------------------------------------------------------------- #
def reset_session():
    for key in ("graph", "thread_id", "phase", "query", "trace", "interrupt", "final"):
        st.session_state.pop(key, None)
    st.session_state.phase = "input"


if "phase" not in st.session_state:
    reset_session()


def render_node_update(node: str, update: dict):
    """Render one node's contribution to the workflow as it streams in."""
    label = NODE_LABELS.get(node, node)
    msgs = update.get("messages") or []
    text = ""
    for m in msgs:
        text += getattr(m, "content", "") + "\n"
    with st.chat_message("assistant"):
        st.markdown(f"**{label}**")
        if text.strip():
            st.markdown(text.strip())
        else:
            st.caption("(working...)")
    st.session_state.trace.append((label, text.strip()))


def stream_until_pause(graph, payload, config):
    """Stream graph execution, rendering each node, until interrupt or end.

    Returns "interrupt" if a human-review pause occurred, else "done".
    """
    for chunk in graph.stream(payload, config=config, stream_mode="updates"):
        if "__interrupt__" in chunk:
            intr = chunk["__interrupt__"][0]
            st.session_state.interrupt = intr.value
            return "interrupt"
        for node, update in chunk.items():
            if isinstance(update, dict):
                render_node_update(node, update)
    return "done"


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
settings = get_settings()
with st.sidebar:
    st.header("TBCG")
    st.caption("TechBro Consulting Group — multi-agent AI consulting firm")
    st.divider()
    st.subheader("Status")
    st.write("LLM token:", "✅ set" if settings.has_llm else "❌ missing (set HF_TOKEN)")
    st.write("Model:", f"`{settings.hf_model}`")
    st.write("Knowledge base:", "✅ Chroma up" if chroma_available() else "⚠️ offline (web-only)")
    st.write("Search:", f"`{settings.search_backend}`")
    st.divider()
    if st.button("🔄 New engagement", use_container_width=True):
        reset_session()
        st.rerun()

st.title("🧠 TBCG — TechBro Consulting Group")
st.caption("Research → Analysis → Strategy → Risk → Critique → Human approval")


# --------------------------------------------------------------------------- #
# Phase: input
# --------------------------------------------------------------------------- #
if st.session_state.phase == "input":
    examples = [
        "Should we expand into the European mattress market?",
        "Should we raise mattress prices by 10% next year?",
        "How can we cut operating costs to sustain positive EBITDA?",
        "Should we acquire Helix Sleep?",
        "Should we prioritize wholesale or DTC for growth?",
    ]
    st.write("**Try an example:**")
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex, use_container_width=True):
            st.session_state.pending_query = ex

    with st.form("query_form"):
        query = st.text_area(
            "Business problem",
            value=st.session_state.get("pending_query", ""),
            placeholder="Describe the strategic question your organization is facing...",
            height=120,
        )
        submitted = st.form_submit_button("Run engagement", type="primary")

    if submitted and query.strip():
        if not settings.has_llm:
            st.error("HF_TOKEN is not set. Add it to your .env before running.")
        else:
            st.session_state.graph = build_graph(human_in_the_loop=True)
            st.session_state.thread_id = uuid.uuid4().hex
            st.session_state.query = query.strip()
            st.session_state.trace = []
            st.session_state.phase = "running"
            st.rerun()


# --------------------------------------------------------------------------- #
# Phase: running (initial stream)
# --------------------------------------------------------------------------- #
elif st.session_state.phase == "running":
    st.info(f"**Engagement:** {st.session_state.query}")
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    with st.spinner("Consultants are working..."):
        result = stream_until_pause(
            st.session_state.graph,
            initial_state(st.session_state.query),
            config,
        )
    st.session_state.phase = "human_review" if result == "interrupt" else "done"
    st.rerun()


# --------------------------------------------------------------------------- #
# Phase: human review (HITL gate)
# --------------------------------------------------------------------------- #
elif st.session_state.phase == "human_review":
    st.info(f"**Engagement:** {st.session_state.query}")
    for label, text in st.session_state.trace:
        with st.chat_message("assistant"):
            st.markdown(f"**{label}**")
            if text:
                st.markdown(text)

    payload = st.session_state.interrupt or {}
    rec = payload.get("recommendation") or {}
    st.divider()
    st.subheader("🙋 Human approval required")
    st.markdown(f"**Recommendation:** {rec.get('recommendation', '—')}")
    if rec.get("confidence") is not None:
        st.caption(f"Confidence: {float(rec['confidence']):.0%}")
    risk = payload.get("risk_assessment") or {}
    if risk:
        st.markdown(f"**Risk level:** {risk.get('risk_level', 'n/a')}")

    feedback = st.text_input("Feedback (required if requesting changes)")
    c1, c2 = st.columns(2)
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    if c1.button("✅ Approve", type="primary", use_container_width=True):
        with st.spinner("Finalizing report..."):
            stream_until_pause(
                st.session_state.graph,
                Command(resume={"approved": True}),
                config,
            )
        st.session_state.phase = "done"
        st.rerun()

    if c2.button("✏️ Request revision", use_container_width=True):
        if not feedback.strip():
            st.warning("Please provide feedback so the team can revise.")
        else:
            with st.spinner("Team is revising..."):
                result = stream_until_pause(
                    st.session_state.graph,
                    Command(resume={"approved": False, "feedback": feedback.strip()}),
                    config,
                )
            st.session_state.phase = "human_review" if result == "interrupt" else "done"
            st.rerun()


# --------------------------------------------------------------------------- #
# Phase: done (final report)
# --------------------------------------------------------------------------- #
elif st.session_state.phase == "done":
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    final = st.session_state.graph.get_state(config).values
    report_md = final.get("final_report", "")

    st.success("Engagement complete.")
    with st.expander("Workflow trace", expanded=False):
        for label, text in st.session_state.trace:
            st.markdown(f"**{label}** — {text}")

    if report_md:
        st.markdown(report_md)
        st.download_button(
            "⬇️ Download PDF report",
            data=export_pdf_bytes(report_md),
            file_name="tbcg_report.pdf",
            mime="application/pdf",
            type="primary",
        )
    else:
        st.warning("No report was produced.")
