import gc
from typing import Generator

import gradio as gr

from logic.rag import build_index_from_file
from retrieval.rag_helper import RAGHelper


def update_button_state(
    index_markdown: str, uploaded_file: str | None = None, rag_helper: RAGHelper | None = None
) -> tuple[gr.Button, RAGHelper | None, str]:
    if not uploaded_file or not rag_helper:
        del rag_helper
        gc.collect()
        return (
            gr.Button(interactive=True, value="Build Index from document"),
            None,
            "No Index built yet.",
        )
    return gr.Button(interactive=False, value="Index built."), rag_helper, index_markdown


def building_button_handler(
    uploaded_file: str | None = None,
) -> Generator[tuple[gr.Button, RAGHelper | None, str], None, None]:
    if not uploaded_file:
        yield (
            gr.Button(interactive=True, value="Build Index from document"),
            None,
            "No documents uploaded. Please upload at least one document.",
        )
    yield (
        gr.Button(interactive=False, value="Building Index ..."),
        None,
        "Please wait ...",
    )

    yield gr.Button(interactive=False, value="Index Built"), *build_index_from_file(
        uploaded_file=uploaded_file
    )
