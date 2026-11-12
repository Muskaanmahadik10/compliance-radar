"""Qdrant-backed document store with hybrid search.

- dense vectors from a fastembed text model (bge-small by default, bge-m3 for the multilingual demo)
- sparse BM25 vectors from fastembed (`Qdrant/bm25`), fused with dense via Reciprocal Rank Fusion inside Qdrant
- optional cross-encoder reranking of the fused top-k
- payload filters (doc_type, topics, supplier_id, jurisdiction) so the agent can scope searches
- delete_document(): removes every chunk of a document (used by the right-to-be-forgotten module)

All models run on CPU via ONNX; nothing leaves the machine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

from radar.config import settings
from radar.infra import get_qdrant
from radar.retrieval.chunking import Chunk

log = logging.getLogger(__name__)
DENSE, SPARSE = "dense", "sparse"


@lru_cache(maxsize=1)
def dense_model() -> TextEmbedding:
    return TextEmbedding(settings.embedding_model)


@lru_cache(maxsize=1)
def sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(settings.sparse_model)


@lru_cache(maxsize=1)
def reranker():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(settings.reranker_model)


@dataclass
class SearchHit:
    chunk_id: str
    doc_id: str
    section: str
    text: str
    score: float
    meta: dict

    @property
    def citation(self) -> str:
        return f"{self.doc_id} §{self.section}"

    def to_dict(self) -> dict:
        return {"citation": self.citation, "doc_id": self.doc_id, "section": self.section, "text": self.text, "score": round(self.score, 4),
                "doc_type": self.meta.get("doc_type"), "title": self.meta.get("title"), "supplier_id": self.meta.get("supplier_id")}


class DocumentStore:
    def __init__(self, client: QdrantClient | None = None, collection: str | None = None):
        self.client = client or get_qdrant()
        self.collection = collection or settings.collection

    # ---- schema ----
    def ensure_collection(self, recreate: bool = False) -> None:
        dim = len(next(iter(dense_model().embed(["probe"]))))
        if recreate and self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
                sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )
            for field, schema in (("doc_id", models.PayloadSchemaType.KEYWORD), ("doc_type", models.PayloadSchemaType.KEYWORD),
                                  ("supplier_id", models.PayloadSchemaType.KEYWORD), ("topics", models.PayloadSchemaType.KEYWORD)):
                try:
                    self.client.create_payload_index(self.collection, field, schema)
                except Exception:  # local mode has no payload indexes; harmless
                    pass

    # ---- write ----
    def upsert_chunks(self, chunks: list[Chunk], batch: int = 64) -> int:
        self.ensure_collection()
        n = 0
        for i in range(0, len(chunks), batch):
            part = chunks[i: i + batch]
            texts = [c.text for c in part]
            dense = list(dense_model().embed(texts))
            sparse = list(sparse_model().embed(texts))
            points = [
                models.PointStruct(
                    id=c.chunk_id,
                    vector={DENSE: d.tolist(), SPARSE: models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())},
                    payload={"doc_id": c.doc_id, "section": c.section, "text": c.text, **{k: v for k, v in c.meta.items() if v is not None}},
                )
                for c, d, s in zip(part, dense, sparse, strict=True)
            ]
            self.client.upsert(self.collection, points)
            n += len(points)
        return n

    def delete_document(self, doc_id: str) -> int:
        before = self.count_document(doc_id)
        self.client.delete(self.collection, points_selector=models.FilterSelector(filter=models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])))
        return before

    def count_document(self, doc_id: str) -> int:
        return self.client.count(self.collection, count_filter=models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]), exact=True).count

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count if self.client.collection_exists(self.collection) else 0

    # ---- read ----
    @staticmethod
    def _filter(doc_type: str | None = None, topics: list[str] | None = None, supplier_id: str | None = None, jurisdiction: str | None = None) -> models.Filter | None:
        must = []
        if doc_type:
            must.append(models.FieldCondition(key="doc_type", match=models.MatchValue(value=doc_type)))
        if supplier_id:
            must.append(models.FieldCondition(key="supplier_id", match=models.MatchValue(value=supplier_id)))
        if jurisdiction:
            must.append(models.FieldCondition(key="jurisdiction", match=models.MatchAny(any=[jurisdiction, "GLOBAL"])))
        if topics:
            must.append(models.FieldCondition(key="topics", match=models.MatchAny(any=topics)))
        return models.Filter(must=must) if must else None

    def search(self, query: str, k: int = 5, mode: str = "hybrid", rerank: bool | None = None, prefetch: int = 20, **filters) -> list[SearchHit]:
        """mode: 'dense' | 'sparse' | 'hybrid' (RRF fusion). rerank defaults to settings.rerank."""
        rerank = settings.rerank if rerank is None else rerank
        flt = self._filter(**filters)
        dq = next(iter(dense_model().embed([query]))).tolist()
        sq = next(iter(sparse_model().embed([query])))
        sqv = models.SparseVector(indices=sq.indices.tolist(), values=sq.values.tolist())
        limit = prefetch if rerank else k
        if mode == "dense":
            res = self.client.query_points(self.collection, query=dq, using=DENSE, limit=limit, query_filter=flt, with_payload=True)
        elif mode == "sparse":
            res = self.client.query_points(self.collection, query=sqv, using=SPARSE, limit=limit, query_filter=flt, with_payload=True)
        else:
            res = self.client.query_points(
                self.collection,
                prefetch=[models.Prefetch(query=dq, using=DENSE, limit=prefetch, filter=flt), models.Prefetch(query=sqv, using=SPARSE, limit=prefetch, filter=flt)],
                query=models.FusionQuery(fusion=models.Fusion.RRF), limit=limit, with_payload=True,
            )
        hits = [SearchHit(str(p.id), p.payload["doc_id"], p.payload["section"], p.payload["text"], float(p.score),
                          {kk: vv for kk, vv in p.payload.items() if kk not in ("doc_id", "section", "text")}) for p in res.points]
        if rerank and hits:
            scores = list(reranker().rerank(query, [h.text for h in hits]))
            for h, s in zip(hits, scores, strict=True):
                h.score = float(s)
            hits.sort(key=lambda h: -h.score)
        return hits[:k]
