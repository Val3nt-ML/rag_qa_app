import gc
import json
import os
from pathlib import Path
import shutil
from typing import Generator
from urllib import response

import gradio as gr
from llama_index.core import PromptTemplate
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceInferenceAPIEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.multi_modal_llms.azure_openai import AzureOpenAIMultiModal

from config import settings
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

        # Initialize models
        progress(0.1, desc="Initializing LLM ...")

        llm: AzureOpenAIMultiModal = AzureOpenAIMultiModal(
            use_azure_ad=False,
            engine=settings.rag.llm_model,
            model=settings.rag.llm_model,
            azure_endpoint=settings.azure.endpoint,
            api_version=settings.azure.api_version,
            api_key=settings.azure.api_key,
        )

        embed_model: AzureOpenAIEmbedding = AzureOpenAIEmbedding(
            model=settings.rag.embed_model,
            azure_endpoint=settings.azure.endpoint,
            api_version=settings.azure.api_version,
            api_key=settings.azure.api_key,
        )

        # embed_model = HuggingFaceInferenceAPIEmbedding(
        #     model_name="google/embeddinggemma-300m",
        #     token=hf_token,
        #     provider="auto",  # this will use the best provider available
        # )

        # Create RAGHelper instance
        rag_helper: RAGHelper = RAGHelper(llm=llm, embed_model=embed_model)

        if uploaded_file:
            # Create temporan ry directory for uploaded files
            temp_dir: Path = Path("temp_uploads")
            temp_dir.mkdir(exist_ok=True)

            # Copy uploaded files to temp directory
            progress(0.2, desc="Copying uploaded file...")
            if os.path.exists(uploaded_file):

                shutil.copy2(uploaded_file, temp_dir)
        # Build index
        if any(temp_dir.iterdir()):
            progress(0.6, desc="Building vector index...")
            rag_helper.build_index(directory_path=str(temp_dir))

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


def generate_question_answer_pairs(
    num_question: int,
    rag_helper: RAGHelper | None = None,
) -> str:
    """
    Apply the RAG helper query engine to answer a user question.

    Args:
        question (str): The user's question

    Returns:
        str: The answer from the RAG system or an error message
    """

    if rag_helper is None or rag_helper._index is None:
        return "❌ Error: No index has been built yet. Please build an index first."

    try:

        dataset_generator = RagDatasetGenerator(
            nodes=rag_helper._nodes[:num_question],
            llm=rag_helper.llm,  # Explicitly providing the LLM here
            num_questions_per_chunk=settings.rag.nb_questions_per_chunk,
            text_question_template=PromptTemplate(
                """
### CONTEXT ###
{context_str}
---------------------
### TOPIC ###
{query_str}
---------------------
### INSTRUCTIONS ###
Based exclusively on the provided CONTEXT, formulate a list of questions related to the TOPIC. Each question must be answerable using only the information given in the text.
""",
            ),
        )

        qa_set = dataset_generator.generate_dataset_from_nodes()
        qa_pairs = [(r.query, r.reference_answer) for r in qa_set.examples]

        temp_dir: Path = Path("temp_qa")
        temp_dir.mkdir(exist_ok=True)

        output_file: str = Path(temp_dir / "qa_pairs_list.json").as_posix()
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
        return output_file

    except Exception as e:
        return f"❌ Error processing your question: {str(e)}"


# Create the Gradio interface
with gr.Blocks(title="Document Q&A Interface") as app:
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

    with gr.Row(visible=True):
        with gr.Column(scale=1):
            num_question = gr.Number(label="Nb Questions to generate", value=5, precision=0)
            generate_btn = gr.Button("Generate and Download Question/Answer pairs")
        with gr.Column(scale=1):
            file_output = gr.File(label="Download your JSON file here")
    # Event handlers -------------------------------------------------------------------
    document_input.change(
        fn=update_button_state,
        inputs=[index_markdown, document_input, rag_helper],
        outputs=[
            build_index_btn,
            rag_helper,
            index_markdown,
        ],
    )

    build_index_btn.click(
        fn=building_button_handler,
        inputs=[document_input],
        outputs=[build_index_btn, rag_helper, index_markdown],
        show_progress="full",
    )

    submit_btn.click(fn=query_documents, inputs=[question_input, rag_helper], outputs=output)

    # QA Pairs Generation Button

    # Connect the button to the function
    generate_btn.click(
        fn=generate_question_answer_pairs,
        inputs=[
            rag_helper,
            num_question,
        ],
        outputs=file_output,
    )

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
    app.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,  # Default Gradio port
        debug=True,
    )
