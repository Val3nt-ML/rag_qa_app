import os
import re

from llama_index.core import (
    PromptTemplate,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.node_parser import (
    MarkdownNodeParser,
    SentenceSplitter,
    SimpleNodeParser,
)
from llama_index.core.schema import BaseNode, Document, ImageDocument, TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
import pymupdf4llm
from pymupdf4llm.llama.pdf_markdown_reader import PDFMarkdownReader
import qdrant_client
from transformers import AutoTokenizer

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
        self.image_similarity_top_k: int = settings.rag.image_similarity_top_k
        # self.max_chunk_size: int = int(
        #     self.llm.metadata.context_window / self.similarity_top_k + 1
        # )
        self.max_chunk_size: int = (
            settings.rag.max_chunk_size
        )  # Arbitrarily set to prevent payload overload issues
        self.chunk_overlap: int = int(self.max_chunk_size / 20)  # 5% overlap
        self.llama_reader: PDFMarkdownReader = pymupdf4llm.LlamaMarkdownReader()
        self.client = qdrant_client.QdrantClient(path="qdrant_mm_db")

    @staticmethod
    def clean_text_node_content(nodes: list[TextNode]) -> list[TextNode]:
        """
        Clean the content of TextNodes by removing markdown image links that can cause
        issues during question generation.

        :param nodes: Description
        :type nodes: list[TextNode]
        :return: Description
        :rtype: list[TextNode]
        """
        # Regex to find and remove Markdown image links: ![*](*)
        markdown_image_regex = r"!\[.*?\]\(.*?\)"

        cleaned_nodes = []
        for node in nodes:
            original_content = node.get_content()
            # Remove all markdown image links from the text
            cleaned_content = re.sub(
                markdown_image_regex, "", original_content, flags=re.DOTALL
            )
            node.set_content(cleaned_content)

            cleaned_nodes.append(node)
        return cleaned_nodes

    def compute_text_nodes(self, text_docs: list[Document]) -> list[BaseNode]:
        splitter: MarkdownNodeParser = MarkdownNodeParser(
            include_metadata=True, include_prev_next_rel=True
        )
        nodes = splitter.get_nodes_from_documents(text_docs)

        tokenizer_encoder = AutoTokenizer.from_pretrained(settings.rag.llm.model)

        # 2nd split to check that none node go beyond the context window of used LLM
        token_aware_splitter = SentenceSplitter(
            chunk_size=self.max_chunk_size,  # Max tokens per chunk
            chunk_overlap=self.chunk_overlap,
            tokenizer=tokenizer_encoder.encode,
        )
        splitted_nodes: list[BaseNode] = token_aware_splitter.get_nodes_from_documents(
            nodes
        )

        return RAGHelper.clean_text_node_content(splitted_nodes)

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
                    write_images=settings.rag.write_images,
                    embed_images=settings.rag.embed_images,
                    image_format=settings.rag.image_format,
                )
            )

        # Add generated images to the documents list
        docs.extend(
            SimpleDirectoryReader(images_path, required_exts=[".png"]).load_data()
        )

        # Compute Text & Image Nodes
        text_docs: list[Document] = [d for d in docs if isinstance(d, Document)]
        text_nodes: list[BaseNode] = self.compute_text_nodes(text_docs)

        image_docs: list[ImageDocument] = [
            d for d in docs if isinstance(d, ImageDocument)
        ]
        img_node_parser: SimpleNodeParser = SimpleNodeParser()
        img_nodes: list[BaseNode] = img_node_parser.get_nodes_from_documents(image_docs)

        self._nodes: list[BaseNode] = text_nodes + img_nodes
        self._text_nodes: list[BaseNode] = text_nodes
        self._image_nodes: list[BaseNode] = img_nodes

    def build_index(self, directory_path: str) -> None:
        """
        Compute Nodes from directory then build corresponding index
        """
        self._compute_nodes(directory_path=directory_path)

        text_store: QdrantVectorStore = QdrantVectorStore(
            client=self.client, collection_name="text_collection"
        )
        image_store: QdrantVectorStore = QdrantVectorStore(
            client=self.client, collection_name="image_collection"
        )
        storage_context = StorageContext.from_defaults(
            vector_store=text_store, image_store=image_store
        )

        self._index = MultiModalVectorStoreIndex(
            self._nodes,
            storage_context=storage_context,
            embed_model=self.embed_model,
            insert_batch_size=512,
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
            image_similarity_top_k=self.image_similarity_top_k,
        )
