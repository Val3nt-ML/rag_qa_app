import os

from llama_index.core import PromptTemplate, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import BaseNode, Document, ImageDocument
import pymupdf4llm
from transformers import AutoTokenizer


class RAGHelper:

    def __init__(
        self,
        llm,
        embed_model,
        text_template: str | None = None,
        hf_token: str | None = None,
        similarity_top_k: int | None = None,
    ):

        self.llm = llm
        self.embed_model = embed_model
        self.text_qa_template: PromptTemplate = (
            PromptTemplate(template=text_template)
            if text_template
            else PromptTemplate(
                "We have provided context information below. \n"
                "---------------------\n"
                "{context_str}"
                "\n---------------------\n"
                "Given this information, please answer the question: {query_str}\n"
            )
        )
        self.hf_token: str | None = hf_token or os.getenv("HF_TOKEN")
        self._index: VectorStoreIndex | None = None
        self.chunk_size: int = 500
        self.chunk_overlap: int = 100
        self.similarity_top_k: int = similarity_top_k or 5
        self.llama_reader = pymupdf4llm.LlamaMarkdownReader()

    def _compute_nodes(self, directory_path: str) -> list[BaseNode]:
        """Load documents from a directory"""

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
                    image_path=directory_path,
                    write_images=True,
                    image_format="png",
                )
            )

        # Add generated images to the documents list
        docs.extend(SimpleDirectoryReader(directory_path, required_exts=[".png"]).load_data())

        # Split Text and Image documents
        text_docs: list[Document] = [d for d in docs if isinstance(d, Document)]
        image_docs: list[ImageDocument] = [d for d in docs if isinstance(d, ImageDocument)]

        # TODO see if I can find a multimodal embedding model to handle both text and images

        splitter: MarkdownNodeParser = MarkdownNodeParser(
            include_metadata=True, include_prev_next_rel=True
        )

        nodes: list[BaseNode] = splitter.get_nodes_from_documents(docs)

        # Check that none node go beyong the context window of used LLM,
        # otherwise chunk it again
        tokenizer = AutoTokenizer.from_pretrained(self.llm.model_name)
        token_aware_splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            tokenizer=tokenizer.encode,
        )
        final_nodes: list[BaseNode] = token_aware_splitter.get_nodes_from_documents(nodes)

        return final_nodes

    def build_index(self, directory_path: str) -> None:
        self._index = VectorStoreIndex(
            nodes=self._compute_nodes(directory_path=directory_path), embed_model=self.embed_model
        )

    def get_query_engine(self) -> BaseQueryEngine:
        if not self._index:
            raise ValueError("Index not built. Call build_index() first.")
        return self._index.as_query_engine(
            llm=self.llm,
            text_qa_template=self.text_qa_template,
            similarity_top_k=self.similarity_top_k,
        )
