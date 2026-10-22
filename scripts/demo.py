"""One-command recruiter demo for the resume RAG pipeline."""

from pathlib import Path

from src.config import Settings
from src.generate import generate_answer
from src.ingest import load_pdf_resumes, split_documents
from src.retrieve import (
    build_faiss_index,
    get_chroma_count,
    get_embeddings,
    init_chroma,
    query_store,
    upsert_and_query_pinecone,
    upsert_chroma,
)

ROOT = Path(__file__).resolve().parents[1]
QUERY = "Python engineer with AWS and machine learning experience"


def load_settings() -> Settings:
    """Load settings with data and Chroma paths anchored at the repository root."""
    env_file = ROOT / ".env"
    return Settings(
        _env_file=env_file if env_file.is_file() else None,
        resumes_dir=(ROOT / "data" / "resumes").resolve(),
        ood_dir=(ROOT / "data" / "ood").resolve(),
        chroma_persist_dir=(ROOT / ".chroma").resolve(),
    )


def _stats(name: str, tokens: list[int]) -> None:
    if not tokens:
        print(f"{name}: no chunks")
        return
    average = sum(tokens) / len(tokens)
    print(
        f"{name}: chunks={len(tokens)} min_words={min(tokens)} "
        f"max_words={max(tokens)} mean_words={average:.1f}"
    )


def _hit_line(store: str, rank: int, candidate: object, score: object) -> str:
    score_text = "" if score is None else str(score)
    return f"{store:<10} {rank:<4} {candidate!s:<22} {score_text}"


def main() -> None:
    settings = load_settings()

    docs = load_pdf_resumes(settings.resumes_dir)
    candidates = sorted({doc.metadata["candidate_id"] for doc in docs})
    print(f"loaded_pages={len(docs)} candidates={len(candidates)}")
    print(candidates)
    assert docs, "expected PDF pages"
    assert all(str(doc.metadata["source_file"]).endswith(".pdf") for doc in docs)
    ood_file = settings.ood_dir / "state_of_the_union.txt"
    assert ood_file.is_file(), f"expected out-of-domain file: {ood_file}"
    assert not any(
        "state_of_the_union" in str(doc.metadata.get("source_file", "")) for doc in docs
    )
    print("state_of_the_union.txt was not indexed")

    char_chunks, char_ids, char_tokens = split_documents(
        docs,
        method="character",
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    recursive_chunks, recursive_ids, recursive_tokens = split_documents(
        docs,
        method="recursive",
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    _stats("character", char_tokens)
    _stats("recursive", recursive_tokens)
    assert len(char_tokens) == len(char_chunks) == len(char_ids)
    assert len(recursive_tokens) == len(recursive_chunks) == len(recursive_ids)
    assert len(recursive_chunks) > 1
    assert all(chunk.page_content.startswith("[Candidate: ") for chunk in char_chunks)
    assert all(chunk.page_content.startswith("[Candidate: ") for chunk in recursive_chunks)

    embeddings = get_embeddings(settings)
    chroma_db = init_chroma(settings, embeddings)
    chroma_db.delete_collection()
    chroma_db = init_chroma(settings, embeddings)

    count_first = upsert_chroma(chroma_db, recursive_chunks, recursive_ids)
    count_second = upsert_chroma(chroma_db, recursive_chunks, recursive_ids)
    print(f"chroma_count_after_first={count_first}")
    print(f"chroma_count_after_second={count_second}")
    print(f"get_chroma_count={get_chroma_count(chroma_db)}")
    assert count_first == count_second == len(recursive_chunks), (
        f"chroma counts diverged: first={count_first} second={count_second} "
        f"chunks={len(recursive_chunks)}"
    )

    chroma_results = query_store(chroma_db, QUERY, k=3)
    faiss_db = build_faiss_index(recursive_chunks, embeddings, recursive_ids)
    faiss_results = query_store(faiss_db, QUERY, k=3)
    assert chroma_results, "expected Chroma hits"
    assert faiss_results, "expected FAISS hits"

    print(f"{'store':<10} {'rank':<4} {'candidate':<22} score")
    for rank, (doc, score) in enumerate(chroma_results, start=1):
        print(_hit_line("chroma", rank, doc.metadata.get("candidate_id"), score))
    for rank, (doc, score) in enumerate(faiss_results, start=1):
        print(_hit_line("faiss", rank, doc.metadata.get("candidate_id"), score))

    pinecone_matches = upsert_and_query_pinecone(
        recursive_chunks,
        recursive_ids,
        embeddings,
        QUERY,
        settings,
        k=3,
    )
    if not settings.has_pinecone:
        assert pinecone_matches == []
    for rank, match in enumerate(pinecone_matches, start=1):
        metadata = match.get("metadata") or {}
        candidate = metadata.get("candidate_id", match.get("id"))
        print(_hit_line("pinecone", rank, candidate, match.get("score")))

    top_docs = [doc for doc, _score in chroma_results]
    answer = generate_answer(QUERY, top_docs, settings)
    print(answer.model_dump_json(indent=2))
    assert answer.query == QUERY
    assert answer.summary
    if not settings.has_openai:
        assert (
            answer.summary
            == "Offline extractive summary (set OPENAI_API_KEY for LLM synthesis)."
        )
        assert len(answer.matches) == len(top_docs)


if __name__ == "__main__":
    main()
