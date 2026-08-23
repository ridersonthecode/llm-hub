"""RAG (Retrieval-Augmented Generation): Text-/Dokumenten-Ablage in Qdrant,
Embeddings über die eigene vLLM-Engine (Modell mit task:"embed", z.B.
Qwen3-Embedding-0.6B - läuft ganz normal im Hot Pool wie jedes andere Modell).

Chunking ist ein simpler, zeichen-basierter Sliding-Window-Splitter (keine
zusätzliche NLP-Bibliothek nötig) - versucht an Absatz-/Satzgrenzen zu brechen
statt hart mitten im Wort."""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Optional

import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from . import process_manager
from .config import Config, get_config

logger = logging.getLogger("vllm_manager.rag")

# Qwen3-Embedding empfiehlt für SUCHANFRAGEN (nicht für Dokumente) ein
# Instruktions-Prefix - laut Modellkarte 1-5% bessere Trefferqualität.
QUERY_INSTRUCTION = (
    "Instruct: Given a search query, retrieve relevant passages that answer the query\n"
    "Query: {query}"
)


class RagNotConfigured(RuntimeError):
    pass


def _require_rag(cfg: Config) -> None:
    if not cfg.rag.enabled or not cfg.rag.embedding_model:
        raise RagNotConfigured(
            "RAG ist nicht aktiviert. In config.json unter \"rag\": \"enabled\": true "
            "und \"embedding_model\" setzen (muss ein registriertes Modell mit "
            "task:\"embed\" sein)."
        )


def _qdrant_client(cfg: Config) -> AsyncQdrantClient:
    return AsyncQdrantClient(host=cfg.rag.qdrant_host, port=cfg.rag.qdrant_port)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            # Lieber an einer Absatz-/Satzgrenze brechen als hart mitten im Wort.
            break_at = text.rfind("\n\n", start, end)
            if break_at == -1 or break_at <= start + chunk_size // 2:
                break_at = text.rfind(". ", start, end)
            if break_at != -1 and break_at > start + chunk_size // 2:
                end = break_at + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embedded eine Liste von Texten über die eigene vLLM-Engine (lädt das
    konfigurierte Embedding-Modell bei Bedarf automatisch, wie jedes andere
    Modell im Hot Pool)."""
    cfg = get_config()
    _require_rag(cfg)
    model = cfg.rag.embedding_model
    status = await process_manager.ensure_loaded(model)
    url = f"http://{cfg.engine_host}:{status['port']}/v1/embeddings"
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, json={"model": model, "input": texts})
        r.raise_for_status()
        data = r.json()
    # Sortiert nach "index", falls die Engine die Reihenfolge nicht garantiert.
    items = sorted(data["data"], key=lambda d: d["index"])
    return [item["embedding"] for item in items]


async def _ensure_collection(client: AsyncQdrantClient, name: str, dim: int) -> None:
    if not await client.collection_exists(name):
        await client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


async def add_text(
    collection: str, text: str, source: str = "text", metadata: Optional[dict] = None
) -> dict:
    cfg = get_config()
    _require_rag(cfg)
    chunks = chunk_text(text, cfg.rag.chunk_size_chars, cfg.rag.chunk_overlap_chars)
    if not chunks:
        return {"document_id": None, "chunks_added": 0}

    vectors = await embed_texts(chunks)
    client = _qdrant_client(cfg)
    try:
        await _ensure_collection(client, collection, len(vectors[0]))
        document_id = uuid.uuid4().hex
        now = time.time()
        points = [
            PointStruct(
                id=uuid.uuid4().hex,
                vector=vec,
                payload={
                    "document_id": document_id,
                    "source": source,
                    "chunk_index": i,
                    "chunk_count": len(chunks),
                    "text": chunk,
                    "added_at": now,
                    **(metadata or {}),
                },
            )
            for i, (chunk, vec) in enumerate(zip(chunks, vectors))
        ]
        await client.upsert(collection_name=collection, points=points)
    finally:
        await client.close()
    return {"document_id": document_id, "chunks_added": len(chunks)}


async def add_file(collection: str, path: str, metadata: Optional[dict] = None) -> dict:
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(
            f"Datei nicht gefunden (Pfad gilt auf dem vLLM-Manager-Server, nicht "
            f"auf deinem Client!): {path}"
        )
    text = extract_text(p)
    meta = {"filename": p.name, **(metadata or {})}
    return await add_text(collection, text, source=str(p), metadata=meta)


async def search(collection: str, query: str, top_k: int = 5) -> list[dict]:
    cfg = get_config()
    _require_rag(cfg)
    vectors = await embed_texts([QUERY_INSTRUCTION.format(query=query)])
    client = _qdrant_client(cfg)
    try:
        if not await client.collection_exists(collection):
            return []
        resp = await client.query_points(
            collection_name=collection,
            query=vectors[0],
            limit=top_k,
            with_payload=True,
        )
    finally:
        await client.close()
    return [
        {
            "score": h.score,
            "text": h.payload.get("text"),
            "source": h.payload.get("source"),
            "document_id": h.payload.get("document_id"),
            "chunk_index": h.payload.get("chunk_index"),
            "chunk_count": h.payload.get("chunk_count"),
            "added_at": h.payload.get("added_at"),
        }
        for h in resp.points
    ]


async def list_collections() -> list[dict]:
    cfg = get_config()
    _require_rag(cfg)
    client = _qdrant_client(cfg)
    try:
        cols = (await client.get_collections()).collections
        out = []
        for c in cols:
            info = await client.get_collection(c.name)
            out.append({"name": c.name, "points_count": info.points_count})
        return out
    finally:
        await client.close()


async def list_documents(collection: str) -> list[dict]:
    """Gruppiert die gespeicherten Chunks nach document_id für eine
    Dokumenten-Ansicht statt einer rohen Chunk-Liste."""
    cfg = get_config()
    _require_rag(cfg)
    client = _qdrant_client(cfg)
    try:
        if not await client.collection_exists(collection):
            return []
        docs: dict[str, dict] = {}
        offset = None
        while True:
            points, offset = await client.scroll(
                collection_name=collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                pl = p.payload or {}
                doc_id = pl.get("document_id", "unbekannt")
                d = docs.setdefault(
                    doc_id,
                    {
                        "document_id": doc_id,
                        "source": pl.get("source"),
                        "filename": pl.get("filename"),
                        "chunk_count": pl.get("chunk_count", 0),
                        "added_at": pl.get("added_at"),
                    },
                )
                d["chunk_count"] = max(d["chunk_count"], pl.get("chunk_count", 0))
            if offset is None:
                break
        return sorted(docs.values(), key=lambda d: d.get("added_at") or 0, reverse=True)
    finally:
        await client.close()


async def delete_document(collection: str, document_id: str) -> dict:
    cfg = get_config()
    _require_rag(cfg)
    client = _qdrant_client(cfg)
    try:
        await client.delete(
            collection_name=collection,
            points_selector=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            ),
        )
    finally:
        await client.close()
    return {"status": "deleted", "document_id": document_id}


async def delete_collection(collection: str) -> dict:
    cfg = get_config()
    _require_rag(cfg)
    client = _qdrant_client(cfg)
    try:
        await client.delete_collection(collection)
    finally:
        await client.close()
    return {"status": "deleted", "collection": collection}
