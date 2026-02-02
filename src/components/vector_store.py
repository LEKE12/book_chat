# src/components/vector_store.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from pymilvus import (
    connections, utility,
    FieldSchema, CollectionSchema, DataType,
    Collection
)

# ----------------------------
# Embedding (local)
# ----------------------------
# Uses sentence-transformers by default (no API).
# If you want Ollama embeddings later, tell me and I’ll swap in an OllamaEmbedder.
from sentence_transformers import SentenceTransformer


@dataclass
class MilvusConfig:
    host: str = "localhost"
    port: str = "19530"
    collection_name: str = "book_chunks"
    dim: int = 384                      # all-MiniLM-L6-v2 output dim
    metric_type: str = "COSINE"         # COSINE is great for text embeddings
    index_type: str = "HNSW"            # fast + good quality
    nlist: int = 1024                   # only used for IVF indexes


class Embedder:
    """
    Local embedder using sentence-transformers.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        # normalize_embeddings=True makes cosine similarity behave nicely
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class MilvusVectorStore:
    def __init__(self, cfg: MilvusConfig, embedder: Optional[Embedder] = None):
        self.cfg = cfg
        self.embedder = embedder or Embedder()
        self.collection: Optional[Collection] = None

    # ----------------------------
    # Connection + collection
    # ----------------------------
    def connect(self) -> None:
        connections.connect(alias="default", host=self.cfg.host, port=self.cfg.port)

    def ensure_collection(self) -> Collection:
        """
        Create collection if not exists, otherwise load it.
        Schema includes:
          - chunk_uid (primary key)
          - doc metadata fields
          - text
          - embedding vector
        """
        name = self.cfg.collection_name

        if utility.has_collection(name):
            col = Collection(name)
            col.load()
            self.collection = col
            return col

        fields = [
            FieldSchema(name="chunk_uid", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=128),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="heading", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="source_path", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.cfg.dim),
        ]

        schema = CollectionSchema(fields, description="Book chunks for RAG")
        col = Collection(name=name, schema=schema)

        # Create index
        if self.cfg.index_type.upper() == "HNSW":
            index_params = {
                "index_type": "HNSW",
                "metric_type": self.cfg.metric_type,
                "params": {"M": 16, "efConstruction": 200},
            }
        else:
            # IVF_FLAT fallback
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": self.cfg.metric_type,
                "params": {"nlist": self.cfg.nlist},
            }

        col.create_index(field_name="embedding", index_params=index_params)
        col.load()
        self.collection = col
        return col

    # ----------------------------
    # Ingestion
    # ----------------------------
    def _iter_jsonl(self, path: Path) -> Iterable[Dict]:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def upsert_chunks_from_jsonl(
        self,
        jsonl_path: str | Path,
        batch_size: int = 64,
    ) -> int:
        """
        Reads your *_chunks.jsonl (from loader.py),
        embeds each chunk text, inserts into Milvus.
        """
        if self.collection is None:
            raise RuntimeError("Call connect() + ensure_collection() first.")

        jsonl_path = Path(jsonl_path).expanduser().resolve()
        if not jsonl_path.exists():
            raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

        inserted = 0
        batch: List[Dict] = []

        def flush(batch_rows: List[Dict]) -> int:
            if not batch_rows:
                return 0

            texts = [r.get("text", "") for r in batch_rows]
            vecs = self.embedder.embed(texts)
            if vecs.shape[1] != self.cfg.dim:
                raise ValueError(
                    f"Embedding dim mismatch. Expected {self.cfg.dim}, got {vecs.shape[1]}.\n"
                    f"Fix by setting MilvusConfig.dim to {vecs.shape[1]}."
                )

            # Milvus insert expects list per field, aligned.
            chunk_uid = [r["chunk_uid"] for r in batch_rows]
            doc_id = [r.get("doc_id", "") for r in batch_rows]
            title = [r.get("title", "") for r in batch_rows]
            heading = [r.get("heading") or "" for r in batch_rows]
            source_path = [r.get("source_path", "") for r in batch_rows]
            text = [r.get("text", "") for r in batch_rows]
            embedding = vecs.tolist()

            self.collection.insert([
                chunk_uid, doc_id, title, heading, source_path, text, embedding
            ])
            return len(batch_rows)

        for row in self._iter_jsonl(jsonl_path):
            # Safety: require chunk_uid and text
            if "chunk_uid" not in row or "text" not in row:
                continue
            batch.append(row)
            if len(batch) >= batch_size:
                inserted += flush(batch)
                batch.clear()

        inserted += flush(batch)
        self.collection.flush()
        return inserted

    def upsert_all_jsonl_in_dir(self, data_dir: str | Path, pattern: str = "*_chunks.jsonl") -> int:
        """
        Ingest all chunk jsonl files (per book) produced by loader.py.
        """
        data_dir = Path(data_dir).expanduser().resolve()
        total = 0
        for p in sorted(data_dir.glob(pattern)):
            n = self.upsert_chunks_from_jsonl(p)
            print(f"✅ Ingested {n} chunks from {p.name}")
            total += n
        return total

    # ----------------------------
    # Search
    # ----------------------------
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.collection is None:
            raise RuntimeError("Call connect() + ensure_collection() first.")

        qvec = self.embedder.embed_one(query).tolist()

        results = self.collection.search(
            data=[qvec],
            anns_field="embedding",
            param={"metric_type": self.cfg.metric_type, "params": {"ef": 64}},
            limit=top_k,
            output_fields=["chunk_uid", "doc_id", "title", "heading", "source_path", "text"],
        )

        hits = []
        for hit in results[0]:
            entity = hit.entity
            hits.append({
                "score": float(hit.score),
                "chunk_uid": entity.get("chunk_uid"),
                "doc_id": entity.get("doc_id"),
                "title": entity.get("title"),
                "heading": entity.get("heading"),
                "source_path": entity.get("source_path"),
                "text": entity.get("text"),
            })
        return hits


def default_store() -> MilvusVectorStore:
    """
    Docker note:
    - If you run inside the app container on the same docker-compose network,
      host should be 'milvus' (service name).
    - If you run locally on your laptop (outside docker), host is 'localhost'.
    """
    host = os.getenv("MILVUS_HOST", "milvus")  # default to docker service name
    port = os.getenv("MILVUS_PORT", "19530")

    cfg = MilvusConfig(
        host=host,
        port=port,
        collection_name=os.getenv("MILVUS_COLLECTION", "book_chunks"),
        dim=int(os.getenv("EMBED_DIM", "384")),
    )
    return MilvusVectorStore(cfg=cfg)
