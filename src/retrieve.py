from pathlib import Path
from typing import Any
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import Settings

def get_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.embedding_model_name)

def init_chroma(settings: Settings, embeddings: HuggingFaceEmbeddings, persist_dir: Path | None = None) -> Chroma:
    target_dir = persist_dir or settings.chroma_persist_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=str(target_dir),
    )

def _scalar_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    clean: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if isinstance(value, bool) or isinstance(value, (int, float, str)):
            clean[str(key)] = value
        elif value is None:
            continue
        else:
            clean[str(key)] = str(value)
    return clean

def _pinecone_metadata(metadata: dict[str, Any], text: str) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in metadata.items():
        name = str(key)
        if isinstance(value, bool) or isinstance(value, (int, float, str)):
            flat[name] = value
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            flat[name] = value
        elif value is None:
            continue
        else:
            flat[name] = str(value)
    flat["text"] = text
    return flat

def _embedder(chroma_db: Chroma):
    return getattr(chroma_db, "_embedding_function", None) or getattr(chroma_db, "embeddings", None)

def _is_duplicate_id_error(exc: BaseException) -> bool:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        name = current.__class__.__name__.lower()
        message = str(current).lower()
        if (
            "duplicate" in name
            or "duplicate" in message
            or "already exist" in message
            or "unique" in message
        ):
            return True
        current = current.__cause__ or current.__context__
    return False

def _collection_upsert(chroma_db: Chroma, chunks: list[Document], ids: list[str]) -> None:
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [_scalar_metadata(chunk.metadata) for chunk in chunks]
    embedder = _embedder(chroma_db)
    embeddings = embedder.embed_documents(texts) if embedder is not None else None
    chroma_db._collection.upsert(
        ids=list(ids),
        embeddings=embeddings,
        metadatas=metadatas,
        documents=texts,
    )

def upsert_chroma(chroma_db: Chroma, chunks: list[Document], ids: list[str]) -> int:
    """Idempotent upsert using deterministic chunk IDs (prevents 105 -> 315 duplication)."""
    if len(chunks) != len(ids):
        raise ValueError("chunks and ids must have the same length")
    storable = [
        Document(page_content=chunk.page_content, metadata=_scalar_metadata(chunk.metadata))
        for chunk in chunks
    ]
    try:
        chroma_db.add_documents(documents=storable, ids=ids)
    except Exception as exc:
        if not _is_duplicate_id_error(exc):
            raise
        # add_documents rejects an existing id; the store upsert path updates in place.
        _collection_upsert(chroma_db, chunks, ids)
    return get_chroma_count(chroma_db)

def get_chroma_count(chroma_db: Chroma) -> int:
    return int(chroma_db._collection.count())

def build_faiss_index(chunks: list[Document], embeddings: HuggingFaceEmbeddings, ids: list[str]) -> FAISS:
    return FAISS.from_documents(documents=chunks, embedding=embeddings, ids=ids)

def query_store(store: Chroma | FAISS, query: str, k: int = 3) -> list[tuple[Document, float]]:
    """Executes a fresh similarity search and returns (Document, score) pairs."""
    return store.similarity_search_with_score(query, k=k)

def upsert_and_query_pinecone(
    chunks: list[Document],
    ids: list[str],
    embeddings: HuggingFaceEmbeddings,
    query: str,
    settings: Settings,
    k: int = 3,
) -> list[dict[str, Any]]:
    """Optional Pinecone comparison helper. Gracefully skips if PINECONE_API_KEY is unset."""
    if not settings.has_pinecone:
        print("Pinecone skipped: PINECONE_API_KEY is not set.")
        return []

    from pinecone import Pinecone

    pc = Pinecone(api_key=settings.pinecone_api_key.get_secret_value())
    index = pc.Index(settings.pinecone_index_name)

    vectors = embeddings.embed_documents([c.page_content for c in chunks])
    records = [
        {
            "id": doc_id,
            "values": vec,
            "metadata": _pinecone_metadata(c.metadata, c.page_content),
        }
        for doc_id, vec, c in zip(ids, vectors, chunks)
    ]
    index.upsert(vectors=records)

    q_vec = embeddings.embed_query(query)
    res = index.query(vector=q_vec, top_k=k, include_metadata=True)
    raw_matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", []) or []
    matches: list[dict[str, Any]] = []
    for match in raw_matches:
        if isinstance(match, dict):
            matches.append(match)
            continue
        metadata = getattr(match, "metadata", None) or {}
        matches.append(
            {
                "id": getattr(match, "id", None),
                "score": getattr(match, "score", None),
                "metadata": dict(metadata),
            }
        )
    return matches
