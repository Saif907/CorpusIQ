import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, SparseIndexParams,
    PointStruct, SparseVector, Filter, FieldCondition, MatchValue,
    PayloadSchemaType, Prefetch, FusionQuery, Fusion
)
from fastembed import SparseTextEmbedding
from app.schemas.rag_schema import RAGChunk, RAGQueryRequest, RAGSearchResult
from app.rag.embeddings import BGEMbeddingEngine



logger = logging.getLogger(__name__)


class QdrantVectorStore:

    def __init__(
        self, 
        collection_name: str = "enron_emails", 
        storage_path = "storage/qdrant_db",
        embedder: Optional[BGEMbeddingEngine] = None
    ):

        self.collection_name = collection_name
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(self.storage_path))
        self.dense_embedder = embedder or BGEMbeddingEngine()
        self.sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")

        self._ensure_collection()


    def _ensure_collection(self):
       
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            logger.info(f"Creating dual Dense + BM25 Qdrant collection '{self.collection_name}'...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                    size=self.dense_embedder.vector_size, 
                    distance=Distance.COSINE
                    )
                },

                sparse_vectors_config={
                    "bm25": SparseVectorParams(index=SparseIndexParams(on_disk=True))
                }
            )
           
            indexed_fields = {
                "thread_id": PayloadSchemaType.KEYWORD,
                "from_address": PayloadSchemaType.KEYWORD,
                "sender_domain": PayloadSchemaType.KEYWORD,
                "is_thread_root": PayloadSchemaType.KEYWORD,
                "thread_position": PayloadSchemaType.INTEGER,
            }
            for field_name, schema_type in indexed_fields.items():
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema_type
                )
            logger.info("Successfully configured Qdrant payload field indexes.")

    
    def ingest_chunks(self, chunks: List[RAGChunk]):

        if not chunks:
            return 
        
        texts_to_embed = [chunk.page_content for chunk in chunks]
        dense_embeddings = self.dense_embedder.embed_texts(texts_to_embed)
        sparse_embeddings = list(self.sparse_embedder.embed(texts_to_embed))

        points = []
        for chunk, d_vector, s_vector in zip(chunks,dense_embeddings,sparse_embeddings):

            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            payload = {
                **chunk.metadata,
                "chunk_id": chunk.chunk_id,
                "page_content": chunk.page_content
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector={
                        "dense": d_vector,
                        "bm25": SparseVector(
                            indices=s_vector.indices.tolist(),
                            values=s_vector.values.tolist()
                        )
                    },
                    payload=payload
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info(f"Added {len(points)} Rag chunks into qdrant store.")

    
    def search(self, request: RAGQueryRequest) -> List[RAGSearchResult]:

        dense_q = self.dense_embedder.embed_query(request.query)
        sparse_q_obj = list(self.sparse_embedder.embed([request.query]))[0]

        sparse_q = SparseVector(
            indices=sparse_q_obj.indices.tolist(), 
            values=sparse_q_obj.values.tolist()
        )

        
        must_conditions = []

        if request.from_address_filter:
            must_conditions.append(
                FieldCondition(key="from_address", match=MatchValue(value=request.from_address_filter))
            )
        if request.thread_id_filter:
            must_conditions.append(
                FieldCondition(key="thread_id", match=MatchValue(value=request.thread_id_filter))
            )
        if request.sender_domain_filter:
            must_conditions.append(
                FieldCondition(key="sender_domain", match=MatchValue(value=request.sender_domain_filter))
            )

        query_filter = Filter(must=must_conditions) if must_conditions else None

        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(query=dense_q, using="dense", filter=query_filter, limit= request.top_k * 2),
                Prefetch(query=sparse_q, using="bm25", filter=query_filter, limit=request.top_k * 2),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=request.top_k
        )

        search_results = []
        for hit in results.points:
            search_results.append(
                RAGSearchResult(
                    score=round(float(hit.score), 4),
                    chunk_id=hit.payload.get("chunk_id", str(hit.id)),
                    page_content=hit.payload.get("page_content", ""),
                    metadata=hit.payload
                )
            )
        
        return search_results