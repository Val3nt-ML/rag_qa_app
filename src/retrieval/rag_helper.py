import os
from pathlib import Path

from llama_index.core import PromptTemplate, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import BaseNode, Document, ImageDocument
import pymupdf4llm
from pymupdf4llm.llama.pdf_markdown_reader import PDFMarkdownReader
import tiktoken

from config import settings


class RAGHelper:

    def __init__(
        self,
        llm,
        embed_model,
        text_template: str | None = None,
    ):

        self.llm = llm
        self.embed_model = embed_model

        self.text_qa_template: PromptTemplate = (
            PromptTemplate(template=text_template)
            if text_template
            else PromptTemplate(
                "# Role\n"
                "You are a specialized technical assistant for design engineers working in an engineering consultancy firm. Your expertise covers mechanical engineering, civil engineering, electrical systems, project management, and technical documentation analysis. Your responsibility is to deliver **precise, technically accurate, and actionable** responses based on engineering documentation and technical standards.\n"
                "\n"
                "# Guidelines\n"
                "- Provide **technically sound and engineering-focused responses** based solely on the provided documentation.\n"
                "- Use appropriate engineering terminology and maintain professional technical language.\n"
                "- When applicable, reference relevant technical standards, codes, or regulations mentioned in the context.\n"
                "- Highlight critical safety considerations, compliance requirements, or design constraints.\n"
                "- Do not make engineering assumptions beyond what is explicitly stated in the documentation.\n"
                "- If calculations or technical specifications are involved, present them clearly and systematically.\n"
                "\n"
                "# Context\n"
                "Here is the retrieved technical documentation and engineering information:\n"
                "{context_str}\n"
                "\n"
                "# Task\n"
                "{query_str}\n"
                "\n"
                "# Response Format\n"
                "- Provide a **comprehensive yet concise technical answer** based exclusively on the provided documentation.\n"
                "- Structure your response with clear sections (e.g., Requirements, Specifications, Compliance, Recommendations) when appropriate.\n"
                "- Include relevant technical details, dimensions, materials, or performance criteria mentioned in the context.\n"
                '- If the documentation lacks sufficient technical information to provide a complete engineering response, state "Insufficient technical documentation available" and specify what additional information would be needed.\n'
                "- Maintain consistency with engineering best practices and respond in the same language as the query.\n"
            )
        )
        self._index: VectorStoreIndex | None = None
        self.similarity_top_k: int = settings.rag.similarity_top_k
        self.chunk_size: int = settings.rag.llm_model_context_window
        self.chunk_overlap: int = int(self.chunk_size / 10)  # 10% overlap
        self.llama_reader: PDFMarkdownReader = pymupdf4llm.LlamaMarkdownReader()

    def _compute_nodes(self, directory_path: str) -> None:
        """Compute nodes from a documents in a directory"""

        images_path: str = directory_path + "/images/"

        docs = []
        # Load PDF documents, convert them to markdown and write images into the same directory
        for file in [
            f
            for f in os.listdir(directory_path)
            if os.path.isfile(os.path.join(directory_path, f))
        ]:
            docs.extend(
                self.llama_reader.load_data(
                    file_path=os.path.join(directory_path, file),
                    image_path=images_path,
                    write_images=True,
                    image_format="png",
                )
            )

        # Add generated images to the documents list
        docs.extend(SimpleDirectoryReader(images_path, required_exts=[".png"]).load_data())

        # Split Text and Image documents
        text_docs: list[Document] = [d for d in docs if isinstance(d, Document)]
        image_docs: list[ImageDocument] = [d for d in docs if isinstance(d, ImageDocument)]

        # TODO see if I can find a multimodal embedding model to handle both text and images

        splitter: MarkdownNodeParser = MarkdownNodeParser(
            include_metadata=True, include_prev_next_rel=True
        )

        nodes: list[BaseNode] = splitter.get_nodes_from_documents(text_docs)

        # Check that none node go beyond the context window of used LLM,
        # otherwise chunk it again

        tokenizer_encoder = tiktoken.encoding_for_model(self.llm.model)
        # TODO Check if it let chunks that are under chunk_size
        token_aware_splitter = SentenceSplitter(
            chunk_size=self.chunk_size,  # Max tokens per chunk
            chunk_overlap=self.chunk_overlap,
            tokenizer=tokenizer_encoder.encode,
        )
        final_nodes: list[BaseNode] = token_aware_splitter.get_nodes_from_documents(nodes)
        self._nodes: list[BaseNode] = final_nodes

    def build_index(self, directory_path: str) -> None:
        """
        Compute Nodes from directory then build corresponding index
        """
        self._compute_nodes(directory_path=directory_path)
        self._index = VectorStoreIndex(
            nodes=self._nodes,
            embed_model=self.embed_model,
            insert_batch_size=1,  # Depends on embedding model
        )

    def get_query_engine(self) -> BaseQueryEngine:
        """
        Compute query engine from index
        """
        if not self._index:
            raise ValueError("Index not built. Call build_index() first.")
        return self._index.as_query_engine(
            llm=self.llm,
            text_qa_template=self.text_qa_template,
            similarity_top_k=self.similarity_top_k,
        )
