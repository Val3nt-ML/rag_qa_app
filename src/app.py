from pathlib import Path

import gradio as gr

from config import settings
from logic.button import building_button_handler, update_button_state
from logic.qa import generate_question_answer_pairs
from logic.rag import query_documents
from retrieval.rag_helper import RAGHelper

EMPTY_ANSWER = (
    "Your answer will appear here.\n\nPrepare a document, then ask your first question."
)
THEME = gr.themes.Base(
    primary_hue="teal",
    secondary_hue="slate",
    neutral_hue="slate",
    font=["system-ui", "sans-serif"],
)
CSS = (Path(__file__).parent / "ui.css").read_text()


def index_controls(helper: RAGHelper | None):
    ready = helper is not None
    return (
        gr.Button(interactive=ready),
        gr.Button(interactive=ready),
        gr.Textbox(interactive=ready),
    )


def clear_outputs():
    return "", EMPTY_ANSWER, None, "Prepare your document to create Q&A cards."


def answer_question(question: str, helper: RAGHelper | None):
    if helper is None:
        return "First, prepare your document in the left panel."
    if not question.strip():
        return "Enter a question about your document."
    answer = query_documents(question, helper)
    if answer.startswith("❌"):
        return "We couldn’t generate an answer. Check that the model is available and try again."
    return answer


def export_questions(count: int, helper: RAGHelper | None):
    result = generate_question_answer_pairs(int(count), helper)
    if result.startswith("❌"):
        return (
            None,
            "**Generation stopped.** Try fewer cards and check that the model is available.",
        )
    return result, "**Your cards are ready.** Download the JSON file below."


with gr.Blocks(title="Document Q&A", analytics_enabled=False) as app:
    rag_helper = gr.State(value=None)
    gr.HTML(
        '<header class="app-header"><div class="brand-mark" aria-hidden="true">D<span>Q</span></div>'
        '<div><p class="eyebrow">YOUR DOCUMENT WORKSPACE</p>'
        '<h1>Document Q&A</h1></div><span class="header-tag">Read. Understand. Remember.</span></header>'
    )
    gr.Markdown(
        "Find answers in your documents. Turn what matters into reusable Q&A cards.",
        elem_classes="intro",
    )
    with gr.Row(equal_height=False, elem_id="workspace"):
        with gr.Column(scale=3, min_width=280, elem_classes="document-panel"):
            gr.Markdown("## 1. Your document\nUpload a PDF to get started.")
            document_input = gr.File(
                label="PDF document", file_types=[".pdf"], type="filepath", height=180
            )
            build_index_btn = gr.Button(
                "Prepare document", variant="primary", interactive=False
            )
            index_markdown = gr.Markdown(
                "**No document yet.** Upload a PDF to get started.",
                elem_id="document-status",
                container=True,
                padding=True,
            )
            with gr.Accordion("How it works", open=False):
                gr.Markdown(
                    "1. **Upload** your PDF.\n2. **Prepare** it to make its content searchable.\n"
                    "3. **Ask a question** or generate Q&A cards.\n\n"
                    "Switching documents? Prepare the new file before continuing.\n\n"
                    "Relevant passages are sent to your configured Hugging Face endpoints."
                )
        with gr.Column(scale=7, min_width=320, elem_classes="reading-panel"), gr.Tabs():
            with gr.Tab("Ask a question", id="questions"):
                gr.Markdown(
                    "## 2. Explore your document\nAsk a specific question to get an answer grounded in your document."
                )
                question_input = gr.Textbox(
                    label="Your question",
                    placeholder="e.g. What are the main requirements in this document?",
                    lines=2,
                    max_lines=5,
                    interactive=False,
                )
                with gr.Row():
                    submit_btn = gr.Button(
                        "Get an answer",
                        variant="primary",
                        interactive=False,
                        scale=3,
                    )
                    clear_btn = gr.Button("Clear", variant="secondary", scale=1)
                with gr.Accordion("Need a starting point?", open=False):
                    gr.Examples(
                        examples=[
                            ["What are the key points in this document?"],
                            ["What requirements must be met?"],
                            ["What are the main steps and expected deliverables?"],
                        ],
                        inputs=[question_input],
                        label="Example questions",
                    )
                gr.Markdown("### Answer")
                output = gr.Markdown(
                    EMPTY_ANSWER,
                    min_height=230,
                    buttons=["copy"],
                    container=True,
                    padding=True,
                    elem_id="answer-panel",
                )
                gr.Markdown(
                    "Check important details against the original document.",
                    elem_classes="quiet-note",
                )
            with gr.Tab("Create Q&A cards", id="export"):
                gr.Markdown(
                    "## 2. Keep the essentials\nGenerate questions and answers to review or reuse what you’ve learned."
                )
                num_question = gr.Number(
                    label="Number of cards",
                    value=5,
                    minimum=1,
                    precision=0,
                    info="For a short document, start with 1 or 2 cards.",
                )
                generate_btn = gr.Button(
                    "Generate cards", variant="primary", interactive=False
                )
                export_status = gr.Markdown(
                    "Prepare your document to create Q&A cards.",
                    elem_id="export-status",
                )
                file_output = gr.File(
                    label="Download Q&A cards (JSON)", interactive=False
                )
                gr.Markdown(
                    "Each card includes a **question**, an **answer**, and the **source passage**. Download the JSON file to use in your own tools."
                )
    gr.HTML(
        '<footer class="app-footer">Document Q&A <span>One document. A clearer understanding.</span></footer>'
    )

    document_input.change(
        update_button_state,
        inputs=[index_markdown, document_input, rag_helper],
        outputs=[build_index_btn, rag_helper, index_markdown],
    ).then(
        index_controls,
        inputs=rag_helper,
        outputs=[submit_btn, generate_btn, question_input],
    ).then(
        clear_outputs,
        outputs=[question_input, output, file_output, export_status],
    )
    build_index_btn.click(
        lambda: gr.File(interactive=False),
        outputs=document_input,
        queue=False,
    ).then(
        building_button_handler,
        inputs=document_input,
        outputs=[build_index_btn, rag_helper, index_markdown],
        show_progress="full",
    ).then(
        index_controls,
        inputs=rag_helper,
        outputs=[submit_btn, generate_btn, question_input],
    ).then(
        lambda: gr.File(interactive=True),
        outputs=document_input,
    )
    gr.on(
        triggers=[submit_btn.click, question_input.submit],
        fn=answer_question,
        inputs=[question_input, rag_helper],
        outputs=output,
        show_progress="full",
        concurrency_limit=1,
    )
    clear_btn.click(
        lambda: ("", EMPTY_ANSWER), outputs=[question_input, output], queue=False
    )
    generate_btn.click(
        export_questions,
        inputs=[num_question, rag_helper],
        outputs=[file_output, export_status],
        show_progress="full",
        concurrency_limit=1,
    )


if __name__ == "__main__":
    app.launch(
        share=settings.app.share,
        server_name=settings.app.server_name,
        server_port=settings.app.server_port,
        debug=settings.app.debug,
        theme=THEME,
        css=CSS,
    )
