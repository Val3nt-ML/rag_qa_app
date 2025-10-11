import os

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.indices.base import BaseIndex
from llama_index.core.schema import Document


class RAGHelper:

    def __init__(self, llm, embed_model, hf_token: str | None = None):

        self.llm = llm
        self.embed_model = embed_model
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self._index: BaseIndex | None = None

    def load_documents(self, directory_path: str) -> list[Document]:
        """Load documents from a directory"""
        reader = SimpleDirectoryReader(directory_path)
        return reader.load_data()

    def build_index(self, documents) -> None:
        Settings.embed_model = self.embed_model
        self._index = VectorStoreIndex.from_documents(documents)

    def get_query_engine(self) -> BaseQueryEngine:
        if not self._index:
            raise ValueError("Index not built. Call build_index() first.")
        return self._index.as_query_engine(llm=self.llm)
