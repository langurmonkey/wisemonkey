"""Session-scoped vector store for document embedding and retrieval.

Uses ChromaDB for persistent storage within each session directory.
Supports PDF (via PyMuPDF) and Markdown/text file ingestion.
Embeddings use OpenAI-compatible API (configurable `base_url`).
"""

import tiktoken
import chromadb
import fitz
from typing import Any
from chromadb.api.types import Metadata

from pathlib import Path
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from agent.config import get_config


class VectorStore:
    """Persistent vector store scoped to a session directory."""

    def __init__(self, session_dir: Path, embedding_model: str | None = None, base_url: str | None = None):
        """Initialize vector store.

        Args:
            session_dir: Path to the session directory.
            embedding_model: Model name for embeddings. Defaults to config.
            base_url: Optional base URL for OpenAI-compatible embedding API. Defaults to config.
        """
        self.session_dir = Path(session_dir)
        self.vectordb_path = self.session_dir / "vectordb"
        self.vectordb_path.mkdir(parents=True, exist_ok=True)

        # Read config
        config = get_config()
        if embedding_model is None:
            embedding_model = config.get("embedding.name", "text-embedding-3-small")
        if base_url is None:
            base_url = config.get("embedding.base_url") or config.get("model.base_url", "")

        # Embedding function
        openai_api_key = ""  # Read from env by openai package
        self.embedding_fn: Any = OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name=embedding_model,
            api_base=base_url if base_url else None,
        )

        # ChromaDB client
        self.client = chromadb.PersistentClient(path=str(self.vectordb_path))
        self.collection = self.client.get_or_create_collection(
            name="documents",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        # Tokenizer for chunking
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def ingest(self, file_path: str) -> int:
        """Ingest a file into the vector store.

        Args:
            file_path: Path to the file to ingest.

        Returns:
            Number of chunks added.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Extract text
        text = self._extract_text(path)
        if not text.strip():
            return 0

        # Chunk text
        chunks = self._chunk_text(text, chunk_size=500, overlap=50)

        # Upsert into ChromaDB
        ids = [f"{path.name}_{i}" for i in range(len(chunks))]
        metadatas: list[Metadata] = [
            {
                "source": str(path.name),
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            for i in range(len(chunks))
        ]

        self.collection.upsert(
            ids=ids,
            documents=chunks,
            metadatas=metadatas,
        )

        return len(chunks)

    def query(self, query_text: str, top_k: int = 3) -> list[dict]:
        """Query the vector store for relevant chunks.

        Args:
            query_text: The query string.
            top_k: Number of results to return.

        Returns:
            List of dicts with 'content', 'source', and 'chunk_index'.
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas"],
        )

        output = []
        if results["documents"] and results["metadatas"]:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                output.append({
                    "content": doc,
                    "source": meta.get("source", "unknown"),
                    "chunk_index": meta.get("chunk_index", 0),
                })

        return output

    def count(self) -> int:
        """Return the number of chunks in the store."""
        return self.collection.count()

    def clear(self):
        """Remove all documents from the store."""
        self.client.delete_collection("documents")
        self.collection = self.client.get_or_create_collection(
            name="documents",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def _extract_text(self, path: Path) -> str:
        """Extract text from a file based on its extension."""
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return self._extract_pdf(path)
        elif suffix in (".md", ".markdown", ".txt", ".text", ".rst"):
            return self._extract_text_file(path)
        else:
            # Try as text file
            return self._extract_text_file(path)

    def _extract_pdf(self, path: Path) -> str:
        """Extract text from a PDF file."""
        text_parts = []
        try:
            doc = fitz.open(str(path))
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from PDF: {e}")

        return "\n".join(text_parts)

    def _extract_text_file(self, path: Path) -> str:
        """Extract text from a text-based file."""
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            raise RuntimeError(f"Failed to read text file: {e}")

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        """Split text into overlapping chunks by token count.

        Args:
            text: The text to chunk.
            chunk_size: Maximum tokens per chunk.
            overlap: Number of overlapping tokens between chunks.

        Returns:
            List of text chunks.
        """
        tokens = self.encoding.encode(text)
        chunks = []

        step = chunk_size - overlap
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i : i + chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text.strip())

        return chunks
