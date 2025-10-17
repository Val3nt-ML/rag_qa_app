import gradio as gr

from config import settings
from logic.button import building_button_handler, update_button_state
from logic.qa import generate_question_answer_pairs
from logic.rag import query_documents

# Create the Gradio interface
with gr.Blocks(title="Document Q&A Interface") as app:

    rag_helper: gr.State = gr.State(value=None)
    gr.Markdown("# 📚 Document Question & Answer Interface")
    gr.Markdown("Upload a document and ask a question about it!")

    gr.Markdown("## 1. Upload a Document & Build an Index")
    # Index Builder section ------------------------------------------------------------
    with gr.Row(visible=True):
        with gr.Column():
            document_input = gr.File(
                label="Upload Document",
                file_types=[".txt", ".pdf", ".docx", ".md"],
                type="filepath",
            )
            build_index_btn = gr.Button("Build Index from document", variant="primary")
        with gr.Column():
            index_markdown = gr.Textbox(
                value="No Index built yet.", lines=2, interactive=False, label="Index Status"
            )

    gr.Markdown("## 2. Ask questions")
    # User Questions -------------------------------------------------------------------
    with gr.Row(visible=True):
        with gr.Column(scale=1):
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="What would you like to know about the document?",
                lines=3,
            )
            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary")
                clear_btn = gr.Button("Clear", variant="secondary")

            # Examples of question
            gr.Examples(
                examples=[
                    [
                        "Quelles sont les exigences minimum requises pour les dispositifs d’isolation?",
                        None,
                    ],
                    [
                        "Fournit un récapitulatif des dispositifs réglementaires applicables dans le document (de préférence sous forme de matrice) ?",
                        None,
                    ],
                    [
                        "A quel phase d’un projet d’isolation doit être remis le rapport d’étanchéité de l’air ? ",
                        None,
                    ],
                    [
                        "Quels sont les différents livrables du management de la qualité en phase DCE?",
                        None,
                    ],
                    ["Quels sont les cibles prioritaires HQE pour les crèches?", None],
                    [
                        "Fournit un récapitulatif des comptages chauffage au 30 mai 2016 au format JSON.",
                        None,
                    ],
                ],
                inputs=[question_input, document_input],
            )
        output = gr.Textbox(label="Response", lines=15, interactive=False)

    # QA Pairs Generation App ----------------------------------------------------------
    gr.Markdown("## 3. Generate (Question, Answer) set and download it")
    with gr.Row(visible=True):
        with gr.Column(scale=1):
            num_question = gr.Number(label="Nb Questions to generate", value=5, precision=0)
            generate_btn = gr.Button("Generate and Download Question/Answer pairs")
        with gr.Column(scale=1):
            file_output = gr.File(label="Download your JSON file here")

    # Events Handling ------------------------------------------------------------------

    # Index Status update
    document_input.change(
        fn=update_button_state,
        inputs=[index_markdown, document_input, rag_helper],
        outputs=[
            build_index_btn,
            rag_helper,
            index_markdown,
        ],
    )

    # Build Index button
    build_index_btn.click(
        fn=building_button_handler,
        inputs=[document_input],
        outputs=[build_index_btn, rag_helper, index_markdown],
        show_progress="full",
    )

    # User Query button
    submit_btn.click(fn=query_documents, inputs=[question_input, rag_helper], outputs=output)
    clear_btn.click(fn=lambda _: None, outputs=question_input)

    # Question/Answer generation button
    generate_btn.click(
        fn=generate_question_answer_pairs,
        inputs=[
            num_question,
            rag_helper,
        ],
        outputs=file_output,
    )


if __name__ == "__main__":
    # Launch the interface
    app.launch(
        share=settings.app.share,
        server_name=settings.app.server_name,
        server_port=settings.app.server_port,  # Default Gradio port
        debug=settings.app.debug,
    )
