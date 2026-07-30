"""Streamlit front-end: one question box, one grounded answer, one trace.

Every claim in an answer cites the exact documentation section it came from, or
the system says "I don't know" — and the per-stage latency and per-model cost of
producing it are shown alongside. Those two things are the point of the project,
so the UI puts both on screen rather than only the prose.

Two modes, one service:

  - **Cached** (default, and the only mode on the public deploy): serves
    precomputed real answers from `demo/examples.json`. Zero LLM calls, no API
    key, and no `index.db` — so it costs nothing to run continuously and
    redistributes none of the source corpus.
  - **Live** (opt-in, local): when both API keys and a built `data/index.db` are
    present, a second box runs the real pipeline against any question. Intended
    for local use; the public host never has the keys to enable it.

Run:  streamlit run app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

from demo.render import IDK, link_citations

ROOT = Path(__file__).resolve().parent
EXAMPLES_FILE = ROOT / "demo" / "examples.json"
DB_FILE = ROOT / "data" / "index.db"

REPO_URL = "https://github.com/Hal-Hanami/tech-docs-rag"

st.set_page_config(page_title="tech-docs-rag", page_icon="📚", layout="centered")


# --- rendering ---------------------------------------------------------------

def render_result(answer: str, sources: list[dict], grounded: bool, trace: dict) -> None:
    """Render one answer + its citations + the request trace (cached or live)."""
    if grounded:
        st.markdown(link_citations(answer, sources))
        st.markdown("**Sources** &nbsp; :green[✓ grounded]")
        for s in sources:
            st.markdown(
                f"<span style='color:#888'>[{s['n']}]</span> "
                f"[{s['section_path']}]({s['url']})",
                unsafe_allow_html=True,
            )
    else:
        st.info(f'**Abstained** — *“{answer}”*')
        st.caption(
            f"The retriever returned {len(sources)} section(s), but none supported an "
            "answer — so the model declined rather than guess. "
            "Hallucination suppression you can see."
        )

    # The M6 trace: what this one request cost and where the time went.
    total_s = trace["total_ms"] / 1000
    cost = trace["total_usd"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Cost (this query)", f"${cost:.4f}")
    c2.metric("Latency", f"{total_s:.1f}s")
    c3.metric("Citations", str(len(sources)))
    with st.expander("🔎 trace — per-stage latency + per-model cost (M6)"):
        stages = " · ".join(f"{name} {ms:.0f}ms" for name, ms in trace["stages"])
        st.markdown(f"**Stages:** {stages}")
        rows = ["| model | tokens | USD |", "|---|---|---|"]
        for model, u in trace["cost_by_model"].items():
            toks = ", ".join(f"{k.replace('_tokens', '')}={v}"
                             for k, v in u.items() if k != "usd")
            rows.append(f"| `{model}` | {toks} | ${u['usd']:.6f} |")
        rows.append(f"| **total** | | **${cost:.4f}** |")
        st.markdown("\n".join(rows))


# --- cached demo (default) ---------------------------------------------------

@st.cache_data
def load_examples() -> dict:
    return json.loads(EXAMPLES_FILE.read_text(encoding="utf-8"))


def cached_demo() -> None:
    data = load_examples()
    questions = [e["question"] for e in data["examples"]]
    choice = st.selectbox("Pick a question", questions, index=0)
    ex = next(e for e in data["examples"] if e["question"] == choice)

    st.markdown(f"### {ex['question']}")
    render_result(ex["answer"], ex["sources"], ex["grounded"], ex["trace"])

    st.caption(
        f"Cached demo · answers precomputed {data['generated_at']} at k={data['k']} "
        f"with **{data['generation_model']}** · retrieval = {data['retrieval']}. "
        "Real measured numbers — no live LLM call, no key, no corpus shipped."
    )


# --- live mode (opt-in, local only) ------------------------------------------

def _load_env() -> None:
    """Populate os.environ from a local .env / st.secrets (keys never committed)."""
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    try:  # Streamlit Cloud secrets, if the operator set any (we don't, for the demo)
        for k in ("ANTHROPIC_API_KEY", "VOYAGE_API_KEY"):
            if k in st.secrets:
                os.environ.setdefault(k, str(st.secrets[k]))
    except Exception:
        pass


def live_available() -> bool:
    """Live mode needs both keys AND a built index — true locally, false on the host."""
    _load_env()
    return bool(os.environ.get("ANTHROPIC_API_KEY")
                and os.environ.get("VOYAGE_API_KEY") and DB_FILE.exists())


def live_mode() -> None:
    st.markdown("Ask anything about the Claude docs. Runs the real pipeline "
                "(retrieve → rerank → grounded generate) at k=3.")
    q = st.text_input("Your question", placeholder="e.g. how do I use the batches API?")
    if not (q and st.button("Ask", type="primary")):
        return
    # Lazy import: only the live path needs the rag package (sqlite-vec/anthropic);
    # the cached deploy runs on streamlit alone.
    from rag import generate as generate_mod
    from rag import observe
    from rag.clients.claude import ClaudeAnswerer
    from rag.clients.voyage import VoyageEmbedder, VoyageReranker

    with st.spinner("retrieving + generating …"):
        embedder, reranker = VoyageEmbedder(), VoyageReranker()
        answerer = ClaudeAnswerer()
        trace = observe.Trace()
        eb, rb = embedder.usage["total_tokens"], reranker.usage["total_tokens"]
        out = generate_mod.answer(q, DB_FILE, embedder, answerer, k=3,
                                  hybrid=True, reranker=reranker, trace=trace)
        trace.add_usage(embedder.model, {"total_tokens": embedder.usage["total_tokens"] - eb})
        trace.add_usage(reranker.model, {"total_tokens": reranker.usage["total_tokens"] - rb})
        trace.add_usage(answerer.model, out["usage"])
        costs = observe.cost_usd(trace.usage_by_model)

    sources = [{"n": i, "section_path": r["section_path"], "url": r["url"]}
               for i, r in enumerate(out["sources"], 1)]
    render_result(out["answer"], sources, out["grounded"], {
        "stages": [[n, s * 1000] for n, s in trace.spans],
        "total_ms": trace.total_seconds * 1000,
        "cost_by_model": {m: {**u, "usd": costs.get(m, 0.0)}
                          for m, u in sorted(trace.usage_by_model.items())},
        "total_usd": costs["total"],
    })


# --- page --------------------------------------------------------------------

st.title("📚 tech-docs-rag")
st.markdown(
    "Source-grounded RAG over the **Claude developer docs**. Ask a question, get "
    "an answer **with a citation to the exact doc section** for every claim — or an "
    f"explicit *“{IDK}”* when the docs don't support one. "
    f"[Code & design notes →]({REPO_URL})"
)

if live_available():
    tab_demo, tab_live = st.tabs(["💡 Cached demo", "🔬 Live (your key)"])
    with tab_demo:
        cached_demo()
    with tab_live:
        live_mode()
else:
    cached_demo()

st.divider()
st.caption(
    "**Why this isn't just a wrapper** — measured retrieval (recall@1 86.4% / "
    "recall@3 100% / MRR 0.932; +25pt recall@1 on hard queries from reranking), "
    "grounded-or-abstain generation (hallucination suppression), and a per-model "
    "cost trace that drove a **−23%/query** optimization (k 5→3) with the "
    "deterministic quality metrics unchanged. Full write-up in the repo README."
)
