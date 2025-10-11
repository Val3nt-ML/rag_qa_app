import gc
import json
import os
from pathlib import Path
from typing import Generator

import gradio as gr
from llama_index.embeddings.huggingface import HuggingFaceInferenceAPIEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

from retrieval.rag_helper import RAGHelper


def update_button_state(
    uploaded_file: str | None = None, rag_helper: RAGHelper | None = None
) -> tuple[gr.Button, RAGHelper | None]:
    if not uploaded_file or not rag_helper:
        if rag_helper:
            del rag_helper
            gc.collect()
        return gr.Button(interactive=True, value="Build Index from document"), None
    return gr.Button(interactive=False, value="Index built."), rag_helper


def building_button_handler(
    uploaded_file: str | None = None,
) -> Generator[gr.Button, RAGHelper | None, str]:
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


def build_index_from_file(
    uploaded_file: str | None = None, progress=gr.Progress(track_tqdm=True)
) -> tuple[RAGHelper | None, str]:
    """
    Initialize a RAGHelper instance with a query engine based on uploaded documents.

    Args:
        uploaded_file (str, optional): Path to uploaded file
        progress: Gradio progress tracker

    Returns:
        RAGHelper: Configured RAGHelper instance with query engine

    Raises:
        Exception: If no documents are provided or configuration fails
    """

    try:
        progress(0, desc="Starting index build...")

        # Load HuggingFace token
        progress(0.1, desc="Loading configuration...")
        with open(Path().resolve() / "hf_token.json") as f:
            hf_token = json.load(f)["HUGGING_FACE_TOKEN"]

        # Initialize models
        progress(0.2, desc="Initializing LLM...")
        llm: HuggingFaceInferenceAPI = HuggingFaceInferenceAPI(
            model_name="deepseek-ai/DeepSeek-V3-0324",
            token=hf_token,
            provider="auto",
            temperature=0.1,
        )

        progress(0.4, desc="Initializing embedding model...")
        embed_model: HuggingFaceInferenceAPIEmbedding = HuggingFaceInferenceAPIEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2", token=hf_token
        )

        # Create RAGHelper instance
        progress(0.5, desc="Creating RAG helper...")
        rag_helper: RAGHelper = RAGHelper(llm=llm, embed_model=embed_model, hf_token=hf_token)

        # Load documents
        progress(0.6, desc="Preparing document directory...")
        documents = []

        if uploaded_file:
            # Create temporary directory for uploaded files
            temp_dir: Path = Path("temp_uploads")
            temp_dir.mkdir(exist_ok=True)

            # Copy uploaded files to temp directory
            progress(0.7, desc="Copying uploaded file...")
            if os.path.exists(uploaded_file):
                import shutil

                shutil.copy2(uploaded_file, temp_dir)

            # Load documents from temp directory
            progress(0.8, desc="Loading documents...")
            if any(temp_dir.iterdir()):
                documents.extend(rag_helper.load_documents(directory_path=str(temp_dir)))

        # Build index
        progress(0.9, desc="Building vector index...")
        rag_helper.build_index(documents)

        progress(1.0, desc="Index build complete!")
        return rag_helper, "Index successfully built ✅"

    except Exception as e:
        raise Exception(f"Failed to initialize RAG helper: {str(e)}")


def query_documents(question: str, rag_helper: RAGHelper | None = None) -> str:
    """
    Apply the RAG helper query engine to answer a user question.

    Args:
        question (str): The user's question

    Returns:
        str: The answer from the RAG system or an error message
    """

    if not question or not question.strip():
        return "❌ Please enter a question."

    if rag_helper is None:
        return "❌ Error: No documents have been uploaded yet. Please upload documents first before asking questions."

    try:
        query_engine = rag_helper.get_query_engine()
        response = query_engine.query(question)
        return f"""**Question:** {question}
🤖 **Answer:** {str(response)}"""

    except Exception as e:
        return f"❌ Error processing your question: {str(e)}"


def clear_inputs():
    """Clear all inputs"""
    return "", None, "", None


# Create the Gradio interface
with gr.Blocks(title="Document Q&A Interface") as demo:
    gr.Markdown("# 📚 Document Question & Answer Interface")
    gr.Markdown("Upload a document and ask a question about it!")

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
        output = gr.Textbox(label="Response", lines=15, max_lines=20, interactive=False)

    rag_helper: gr.State = gr.State(value=None)

    # Event handlers -------------------------------------------------------------------
    document_input.change(
        fn=update_button_state,
        inputs=[document_input, rag_helper],
        outputs=[build_index_btn, rag_helper],  # 🔑 Key: The button itself is the output!
    )

    build_index_btn.click(
        fn=building_button_handler,
        inputs=[document_input],
        outputs=[build_index_btn, rag_helper, index_markdown],
        show_progress="full",
    )

    submit_btn.click(fn=query_documents, inputs=[question_input, rag_helper], outputs=output)

    # clear_btn.click(fn=clear_inputs, outputs=[question_input, document_input, output])

    # Examples
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

if __name__ == "__main__":
    # Launch the interface
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,  # Default Gradio port
        debug=True,
    )
