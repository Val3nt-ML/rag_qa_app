import os
from pathlib import Path
import shutil

import gradio as gr
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.multi_modal_llms.azure_openai import AzureOpenAIMultiModal

from config import settings
from retrieval.rag_helper import RAGHelper


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
            engine=settings.rag.llm.model_name,
            model=settings.rag.llm.model_name,
            azure_endpoint=settings.azure.endpoint,
            api_version=settings.azure.api_version,
            api_key=settings.azure.api_key,
            temperature=settings.rag.llm.temperature,
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
