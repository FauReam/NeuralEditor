"""Hybrid memory: short-term window + long-term RAG."""

import uuid
from pathlib import Path
from typing import Any

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from sentence_transformers import SentenceTransformer
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


class MemorySystem:
    """Short-term conversation buffer + long-term vector retrieval."""

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        vector_db_path: str = "data/memories/chroma",
        short_term_turns: int = 10,
        long_term_top_k: int = 3,
        similarity_threshold: float = 0.75,
    ):
        self.short_term: list[dict[str, str]] = []
        self.short_term_turns = short_term_turns
        self.long_term_top_k = long_term_top_k
        self.similarity_threshold = similarity_threshold

        self._embedding_model_name = embedding_model
        self._embedder: Any = None
        self._chroma_client: Any = None
        self._collection: Any = None
        self._vector_db_path = vector_db_path

    def _init_vector_db(self) -> None:
        if not CHROMA_AVAILABLE:
            return
        if self._chroma_client is not None:
            return

        try:
            self._embedder = SentenceTransformer(self._embedding_model_name, device="cpu")
        except Exception as e:
            print(f"  [!] Embedding 模型加载失败（向量记忆不可用）: {e}")
            self._embedder = None

        Path(self._vector_db_path).parent.mkdir(parents=True, exist_ok=True)
        self._chroma_client = chromadb.Client(
            ChromaSettings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self._vector_db_path,
            )
        )
        self._collection = self._chroma_client.get_or_create_collection(
            name="long_term_memory",
            metadata={"hnsw:space": "cosine"},
        )

    def add_turn(self, role: str, content: str) -> None:
        """Add a conversation turn to short-term memory."""
        self.short_term.append({"role": role, "content": content})
        if len(self.short_term) > self.short_term_turns * 2:
            self.short_term = self.short_term[-self.short_term_turns * 2:]

    def add_long_term(self, text: str, memory_id: str | None = None) -> None:
        """Embed and store a long-term memory."""
        if not CHROMA_AVAILABLE:
            return
        self._init_vector_db()

        mid = memory_id or str(uuid.uuid4())
        if self._embedder is None:
            return
        embedding = self._embedder.encode(text).tolist()
        self._collection.add(
            ids=[mid],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{"source": "conversation"}],
        )

    def retrieve_relevant(self, query: str) -> list[str]:
        """Query long-term memory for relevant context."""
        if not CHROMA_AVAILABLE or self._collection is None:
            return []

        if self._embedder is None:
            return []
        embedding = self._embedder.encode(query).tolist()
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=self.long_term_top_k,
        )

        docs = results["documents"][0]
        distances = results["distances"][0]
        # Chroma returns cosine distance (0=identical)
        return [
            doc for doc, dist in zip(docs, distances)
            if (1 - dist) >= self.similarity_threshold
        ]

    def get_short_term_context(self) -> list[dict[str, str]]:
        return self.short_term.copy()

    def summarize_and_store(self) -> None:
        """Placeholder: summarize old short-term into long-term."""
        pass