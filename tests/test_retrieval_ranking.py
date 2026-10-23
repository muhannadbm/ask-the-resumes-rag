"""Real-model Chroma lock for the demo query and the unrelated-query distance."""

from pathlib import Path

import pytest
from langchain_core.documents import Document

from scripts.demo import QUERY
from scripts.seed_sample_pdfs import OOD_TEXT, PROFILES, build_body, write_pdf
from src.config import Settings
from src.ingest import load_pdf_resumes, split_documents
from src.retrieve import get_embeddings, init_chroma, query_store, upsert_chroma

REACT_QUERY = "React"
KUBERNETES_QUERY = "Kubernetes"
UNRELATED_QUERY = OOD_TEXT
RECORDED_UNRELATED_DISTANCE = 0.9903925061225891

_NAMES = {Path(profile["file_name"]).stem: profile["name"] for profile in PROFILES}


def _names(rows: list[tuple[Document, float]]) -> list[str]:
    return [_NAMES[doc.metadata["candidate_id"]] for doc, _score in rows]


def _distances(rows: list[tuple[Document, float]]) -> list[float]:
    return [float(score) for _doc, score in rows]


@pytest.fixture(scope="module")
def chroma_index(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("retrieval-ranking")
    resumes = root / "resumes"
    resumes.mkdir()
    for profile in PROFILES:
        write_pdf(resumes / profile["file_name"], build_body(profile))

    settings = Settings(
        _env_file=None,
        resumes_dir=resumes,
        ood_dir=root / "ood",
        chroma_persist_dir=root / "chroma",
        chroma_collection_name="retrieval_ranking",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        pinecone_api_key=None,
        openai_api_key=None,
    )
    pages = load_pdf_resumes(resumes)
    chunks, ids, _token_counts = split_documents(
        pages,
        method="recursive",
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    embeddings = get_embeddings(settings)
    store = init_chroma(settings, embeddings, persist_dir=settings.chroma_persist_dir)
    upsert_chroma(store, chunks, ids)
    yield store
    store.delete_collection()
    from chromadb.api.client import SharedSystemClient

    SharedSystemClient.clear_system_cache()


def test_unrelated_query_records_closest_distance(chroma_index) -> None:
    rows = query_store(chroma_index, UNRELATED_QUERY, k=4)
    distances = _distances(rows)
    assert distances == sorted(distances)
    assert distances[0] == min(distances)
    assert distances[0] == RECORDED_UNRELATED_DISTANCE


def test_react_and_kubernetes_queries_return_closer_hits_first(chroma_index) -> None:
    for query in (REACT_QUERY, KUBERNETES_QUERY):
        rows = query_store(chroma_index, query, k=4)
        distances = _distances(rows)
        assert rows
        assert distances == sorted(distances)


def test_demo_query_ranking(chroma_index) -> None:
    rows = query_store(chroma_index, QUERY, k=4)
    names = _names(rows)
    assert (
        len(names) == 4
        and names[0] == "Blair Chen"
        and len(set(names)) == len(names)
        and "Alex Rivera" in names[:3]
    ), names
