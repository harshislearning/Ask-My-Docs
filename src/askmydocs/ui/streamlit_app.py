"""Streamlit front end.

    streamlit run src/askmydocs/ui/streamlit_app.py

A thin client over the HTTP API. All the logic worth testing lives in
``api_client``, ``formatting`` and ``theme``; this module is layout and state,
because a Streamlit script re-runs top to bottom on every interaction and
anything non-trivial defined here would be hard to test and easy to break
unnoticed.

The API address comes from the ``ASKMYDOCS_API_URL`` environment variable. It
is not a form field: it changes roughly never, and a text input for it puts a
piece of deployment configuration in front of someone who came to ask a
question.
"""

from __future__ import annotations

import os
import time
from typing import Any

import streamlit as st

from askmydocs.ui.api_client import DEFAULT_BASE_URL, ApiError, AskMyDocsClient
from askmydocs.ui.formatting import (
    cited_numbers,
    format_duration,
    highlight_citations,
    source_label,
    verification_badges,
)
from askmydocs.ui.theme import CSS

#: How long to wait for an ingestion job before telling the user to check back.
JOB_POLL_SECONDS = 2
JOB_TIMEOUT_SECONDS = 900

EXAMPLE_PROMPTS = [
    "What is the default timeout?",
    "How does the rollback process work?",
    "Which parameters can be configured?",
]


def api_url() -> str:
    return os.environ.get("ASKMYDOCS_API_URL", DEFAULT_BASE_URL)


def main() -> None:
    st.set_page_config(
        page_title="Ask My Docs",
        page_icon="📄",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    st.session_state.setdefault("history", [])

    client = AskMyDocsClient(api_url())
    health = _sidebar(client)

    st.markdown(
        '<div class="amd-header">'
        '<div class="amd-title">Ask My <span>Docs</span></div>'
        '<div class="amd-subtitle">Answers from your documents, with every claim cited.</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    ask_tab, documents_tab = st.tabs(["Ask", "Documents"])
    with ask_tab:
        _ask_panel(client, health)
    with documents_tab:
        _documents_panel(client)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------


def _sidebar(client: AskMyDocsClient) -> dict[str, Any] | None:
    with st.sidebar:
        health = _status(client)

        st.markdown('<div class="amd-label">Retrieval</div>', unsafe_allow_html=True)
        st.session_state["top_k"] = st.slider(
            "Sources per answer",
            min_value=1,
            max_value=15,
            value=6,
            help="How many passages the model is shown after reranking.",
        )
        st.session_state["rerank"] = st.toggle(
            "Cross-encoder reranking",
            value=True,
            help="Turn off to see what the retrievers alone return.",
        )

        if st.session_state["history"]:
            st.markdown("")
            if st.button("Clear history", use_container_width=True):
                st.session_state["history"] = []
                st.rerun()

        if health and health.get("embedding_model"):
            st.markdown(
                f'<div class="amd-stat" style="margin-top:1.5rem">'
                f"Embeddings · {health['embedding_model'].split('/')[-1]}<br>"
                f"Answers · llama-3.3-70b"
                "</div>",
                unsafe_allow_html=True,
            )
        return health


def _status(client: AskMyDocsClient) -> dict[str, Any] | None:
    """A single status pill, so the state of the backend is legible at a glance."""
    try:
        health = client.health()
    except ApiError as exc:
        st.markdown(
            '<div class="amd-status"><span class="amd-dot offline"></span>'
            "Backend offline</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="amd-stat">{exc.message}</div>', unsafe_allow_html=True)
        return None

    ready = health["status"] == "ok"
    st.markdown(
        f'<div class="amd-status">'
        f'<span class="amd-dot {"ready" if ready else "degraded"}"></span>'
        f'{"Ready" if ready else "Needs attention"}</div>',
        unsafe_allow_html=True,
    )

    if health.get("index_built"):
        detail = (
            f"{health.get('chunk_count', 0)} passages from "
            f"{health.get('document_count', 0)} documents"
        )
    else:
        detail = "No documents indexed yet."
    st.markdown(f'<div class="amd-stat">{detail}</div>', unsafe_allow_html=True)
    return health


# --------------------------------------------------------------------------
# Ask
# --------------------------------------------------------------------------


def _ask_panel(client: AskMyDocsClient, health: dict[str, Any] | None) -> None:
    if health is None:
        _empty("🔌", "Backend not running", "Start the API, then reload this page.")
        return

    if not health.get("index_built"):
        _empty("📥", "No documents yet", "Add PDFs on the Documents tab to get started.")
        return

    question = st.chat_input("Ask a question about your documents")
    if question:
        _answer(client, question)

    if not st.session_state["history"]:
        _empty(
            "💬",
            "Ask anything about your documents",
            " · ".join(f"“{prompt}”" for prompt in EXAMPLE_PROMPTS),
        )
        return

    # Newest first: the answer just asked for should never be below a scroll.
    for entry in reversed(st.session_state["history"]):
        _render_answer(entry)


def _answer(client: AskMyDocsClient, question: str) -> None:
    with st.spinner("Searching your documents…"):
        try:
            response = client.query(
                question,
                top_k=st.session_state.get("top_k"),
                rerank=st.session_state.get("rerank"),
            )
        except ApiError as exc:
            st.error(exc.message)
            if exc.is_not_ready:
                st.info("Run ingestion on the **Documents** tab.")
            return

    st.session_state["history"].append(response)


def _render_answer(response: dict[str, Any]) -> None:
    refused = response.get("refused", False)
    body = (
        response["answer"] if refused else highlight_citations(response["answer"])
    )

    st.markdown(
        f'<div class="amd-card{" refusal" if refused else ""}">'
        f'<div class="amd-question">{_escape(response["question"])}</div>'
        f'<div class="amd-answer">{body}</div>'
        f"{_badge_row(response)}"
        "</div>",
        unsafe_allow_html=True,
    )

    for issue in (response.get("verification") or {}).get("issues", []):
        st.caption(f"⚠ {issue['detail']}")
        if issue.get("sentence"):
            st.caption(f"> {issue['sentence']}")

    _render_sources(response)

    st.markdown(
        '<div class="amd-meta">'
        f"<span>{format_duration(response.get('total_ms', 0))}</span>"
        f"<span>{response.get('prompt_tokens') or '?'} in · "
        f"{response.get('completion_tokens') or '?'} out</span>"
        f"<span>{response.get('model', '')}</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def _badge_row(response: dict[str, Any]) -> str:
    badges = verification_badges(response.get("verification"))
    pills = "".join(
        f'<span class="amd-badge {tone}">{_escape(label)}</span>' for label, tone in badges
    )
    return f'<div class="amd-badges">{pills}</div>'


def _render_sources(response: dict[str, Any]) -> None:
    sources = response.get("sources") or []
    if not sources:
        return

    used = cited_numbers(response.get("answer", ""))
    st.markdown(
        f'<div class="amd-label">Sources · {len(used)} of {len(sources)} cited</div>',
        unsafe_allow_html=True,
    )

    for source in sources:
        was_cited = source["number"] in used
        # Cited sources open by default - they are the evidence for what was
        # just claimed. Uncited ones stay collapsed but visible, so it is
        # obvious what the model was shown and chose not to use.
        with st.expander(
            f"{'✓' if was_cited else '○'}  {source_label(source)}", expanded=was_cited
        ):
            if source.get("rerank_score") is not None:
                st.caption(f"relevance {source['rerank_score']:+.2f}")
            st.write(source.get("text") or "_(text not requested)_")


def _empty(icon: str, title: str, hint: str) -> None:
    st.markdown(
        f'<div class="amd-empty"><div class="amd-empty-icon">{icon}</div>'
        f'<div class="amd-empty-title">{title}</div>'
        f'<div class="amd-empty-hint">{hint}</div></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------


def _documents_panel(client: AskMyDocsClient) -> None:
    uploads = st.file_uploader(
        "Add PDFs", type=["pdf"], accept_multiple_files=True, key="uploads"
    )

    left, right = st.columns(2)
    with left:
        if st.button("Upload", disabled=not uploads, use_container_width=True):
            _upload(client, uploads or [])
    with right:
        if st.button("Ingest & index", type="primary", use_container_width=True):
            _run_ingest(client, force=st.session_state.get("force_reparse", False))

    st.checkbox(
        "Re-parse everything", key="force_reparse", help="Ignore the unchanged-file cache"
    )

    st.markdown('<div class="amd-label">Indexed documents</div>', unsafe_allow_html=True)
    _document_table(client)


def _upload(client: AskMyDocsClient, uploads: list[Any]) -> None:
    try:
        result = client.upload([(f.name, f.getvalue()) for f in uploads])
    except ApiError as exc:
        st.error(exc.message)
        return

    if result["uploaded"]:
        st.success(f"Uploaded {', '.join(result['uploaded'])} — now press **Ingest & index**.")
    if result["rejected"]:
        st.warning(f"Rejected: {', '.join(result['rejected'])}")


def _run_ingest(client: AskMyDocsClient, *, force: bool) -> None:
    try:
        job = client.ingest(force=force, rebuild_index=True)
    except ApiError as exc:
        st.error(exc.message)
        return

    status = st.status("Reading documents…", expanded=True)
    deadline = time.monotonic() + JOB_TIMEOUT_SECONDS

    while time.monotonic() < deadline:
        time.sleep(JOB_POLL_SECONDS)
        try:
            job = client.job(job["id"])
        except ApiError as exc:
            status.update(label=exc.message, state="error")
            return

        if job["status"] == "succeeded":
            result = job.get("result") or {}
            status.update(
                label=f"Indexed {result.get('chunks_total', 0)} passages "
                f"from {result.get('documents_ok', 0)} documents",
                state="complete",
            )
            st.rerun()
        if job["status"] == "failed":
            status.update(label="Ingestion failed", state="error")
            st.error(job.get("error") or "unknown error")
            return
        status.write(f"status: {job['status']}")

    status.update(label="Still running — check back shortly", state="error")


def _document_table(client: AskMyDocsClient) -> None:
    try:
        payload = client.documents()
    except ApiError as exc:
        st.error(exc.message)
        return

    documents = payload.get("documents") or []
    if not documents:
        _empty("📄", "Nothing indexed yet", "Upload PDFs above, then press Ingest & index.")
        return

    st.dataframe(
        [
            {
                "File": doc["source_file"],
                "Status": doc["status"],
                "Pages": doc["page_count"],
                "Passages": doc["chunk_count"],
                # Why a document was skipped is the whole point of this table.
                "Note": doc.get("reason") or "",
            }
            for doc in documents
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"{payload.get('total_chunks', 0)} passages indexed")


def _escape(text: str) -> str:
    import html

    return html.escape(text)


if __name__ == "__main__":
    main()
