from collections.abc import Generator

import gradio as gr

from logic.rag import build_index_from_file
from retrieval.rag_helper import RAGHelper


def update_button_state(
    index_markdown: str,
    uploaded_file: str | None = None,
    rag_helper: RAGHelper | None = None,
) -> tuple[gr.Button, None, str]:
    """A changed document always invalidates the previous index."""
    return (
        gr.Button(interactive=bool(uploaded_file), value="Prepare document"),
        None,
        "**Document uploaded.** Select “Prepare document” to continue."
        if uploaded_file
        else "**No document yet.** Upload a PDF to get started.",
    )


def building_button_handler(
    uploaded_file: str | None = None,
) -> Generator[tuple[gr.Button, RAGHelper | None, str]]:
    if not uploaded_file:
        yield gr.Button(interactive=False), None, "Upload a PDF document first."
        return
    yield (
        gr.Button(interactive=False, value="Preparing document…"),
        None,
        "**Preparing document…** Reading and indexing your document. This may take a moment.",
    )
    try:
        helper, _ = build_index_from_file(uploaded_file=uploaded_file)
        if helper is None:
            raise ValueError("No index returned")
    except Exception:  # noqa: BLE001 - Recover UI controls after provider failures.
        yield (
            gr.Button(interactive=True, value="Try again"),
            None,
            "**Preparation failed.** Check your PDF and model availability, then try again.",
        )
        return
    yield (
        gr.Button(interactive=False, value="Document ready"),
        helper,
        "**Your document is ready.** Ask a question or create Q&A cards.",
    )
