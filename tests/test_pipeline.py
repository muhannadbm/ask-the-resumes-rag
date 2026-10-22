import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import Settings
from src.generate import RAGAnswer, generate_answer
from src.ingest import load_pdf_resumes, split_documents
from src.retrieve import (
    get_chroma_count,
    init_chroma,
    upsert_and_query_pinecone,
    upsert_chroma,
)

ROOT = Path(__file__).resolve().parents[1]
RESUME_DIR = ROOT / "data" / "resumes"
OOD_FILE = ROOT / "data" / "ood" / "state_of_the_union.txt"


class HashEmbeddings(Embeddings):
    """Deterministic stand-in so storage tests do not download a model."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        import hashlib

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values: list[float] = []
        while len(values) < self.dim:
            digest = hashlib.sha256(digest).digest()
            values.extend(byte / 255.0 for byte in digest)
        return values[: self.dim]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def _isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **overrides) -> Settings:
    monkeypatch.chdir(tmp_path)
    for name in ("PINECONE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return Settings(_env_file=None, pinecone_api_key=None, openai_api_key=None, **overrides)


def test_settings_defaults_without_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _isolated_settings(tmp_path, monkeypatch)
    assert settings.has_pinecone is False
    assert settings.has_openai is False


def test_pdf_loader_ignores_txt_files(tmp_path: Path) -> None:
    pdfs = sorted(RESUME_DIR.glob("*.pdf"))
    assert len(pdfs) == 9
    assert OOD_FILE.is_file()

    corpus = tmp_path / "resumes"
    corpus.mkdir()
    for pdf in pdfs:
        shutil.copy(pdf, corpus / pdf.name)
    shutil.copy(OOD_FILE, corpus / OOD_FILE.name)
    (corpus / "notes.md").write_text("not a resume", encoding="utf-8")

    docs = load_pdf_resumes(corpus)
    assert docs
    assert all(doc.metadata["source_file"].endswith(".pdf") for doc in docs)
    assert all(doc.metadata["candidate_id"] for doc in docs)

    seen: list[str] = []
    for doc in docs:
        source_name = doc.metadata["source_file"]
        if not seen or seen[-1] != source_name:
            seen.append(source_name)
        assert doc.metadata["candidate_id"] == Path(source_name).stem
    assert seen == sorted(pdf.name for pdf in pdfs)

    missing = tmp_path / "missing-resumes"
    with pytest.raises(FileNotFoundError):
        load_pdf_resumes(missing)


def test_chunk_enrichment_and_token_counts(tmp_path: Path) -> None:
    pdf = sorted(RESUME_DIR.glob("*.pdf"))[0]
    corpus = tmp_path / "resumes"
    corpus.mkdir()
    shutil.copy(pdf, corpus / pdf.name)

    pages = load_pdf_resumes(corpus)
    assert len(pages) >= 2

    chunks, ids, recursive_tokens = split_documents(pages, method="recursive")
    assert len(chunks) > 1
    assert all(chunk.page_content.startswith("[Candidate: ") for chunk in chunks)
    assert len(recursive_tokens) == len(chunks) == len(ids)
    assert recursive_tokens == [len(chunk.page_content.split()) for chunk in chunks]
    assert ids == [f"{pdf.stem}::recursive::chunk_{index}" for index in range(len(chunks))]

    anonymous = Document(page_content="Distributed systems experience. " * 80, metadata={})
    unknown_chunks, unknown_ids, unknown_counts = split_documents([anonymous], method="recursive")
    assert unknown_chunks
    assert all(chunk.page_content.startswith("[Candidate: unknown]") for chunk in unknown_chunks)
    assert all(chunk_id.startswith("unknown::recursive::chunk_") for chunk_id in unknown_ids)
    assert len(unknown_chunks) == len(unknown_ids) == len(unknown_counts)


def test_chroma_idempotent_upsert(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = sorted(RESUME_DIR.glob("*.pdf"))[0]
    corpus = tmp_path / "resumes"
    corpus.mkdir()
    shutil.copy(pdf, corpus / pdf.name)
    pages = load_pdf_resumes(corpus)
    chunks, ids, _counts = split_documents(pages, method="recursive", chunk_size=1000, chunk_overlap=100)
    assert len(chunks) == len(ids)

    settings = _isolated_settings(
        tmp_path,
        monkeypatch,
        chroma_persist_dir=tmp_path / "chroma",
        chroma_collection_name="resume_upsert_test",
    )
    chroma_db = init_chroma(settings, HashEmbeddings(), persist_dir=settings.chroma_persist_dir)

    for _ in range(3):
        count = upsert_chroma(chroma_db, chunks, ids)
        assert count == len(chunks)
    assert get_chroma_count(chroma_db) == len(chunks)

    chunks[0].page_content += "\nUPDATED-MARKER"
    count = upsert_chroma(chroma_db, chunks, ids)
    assert count == len(chunks)
    stored = chroma_db._collection.get(ids=[ids[0]], include=["documents"])
    assert stored["documents"]
    assert "UPDATED-MARKER" in stored["documents"][0]


def test_pinecone_absent_returns_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _isolated_settings(tmp_path, monkeypatch)
    result = upsert_and_query_pinecone(
        chunks=[],
        ids=[],
        embeddings=MagicMock(),
        query="python",
        settings=settings,
    )
    assert result == []


def test_generate_answer_offline_extractive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _isolated_settings(tmp_path, monkeypatch)
    docs = [
        Document(
            page_content="[Candidate: 01_alex_rivera]\nPython engineer with AWS experience.",
            metadata={"candidate_id": "01_alex_rivera", "chunk_id": "01_alex_rivera::recursive::chunk_0"},
        ),
        Document(
            page_content="[Candidate: 02_blair_chen]\nMachine learning engineer.",
            metadata={"candidate_id": "02_blair_chen", "chunk_id": "02_blair_chen::recursive::chunk_0"},
        ),
    ]

    with patch("src.generate.OpenAI") as openai_cls:
        answer = generate_answer("python aws", docs, settings, client=None)
        openai_cls.assert_not_called()

    assert isinstance(answer, RAGAnswer)
    assert "extractive" in answer.summary.lower()
    assert len(answer.matches) == len(docs)
    assert [match.candidate_id for match in answer.matches] == ["01_alex_rivera", "02_blair_chen"]
    assert all(match.evidence_quote for match in answer.matches)

    empty = generate_answer("unrelated", [], settings, client=None)
    assert empty.matches == []


def test_generate_answer_structured_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _isolated_settings(tmp_path, monkeypatch)
    docs = [
        Document(
            page_content="[Candidate: 01_alex_rivera]\nPython",
            metadata={"candidate_id": "01_alex_rivera", "chunk_id": "01_alex_rivera::recursive::chunk_0"},
        )
    ]
    expected = RAGAnswer(query="python", summary="mocked synthesis", matches=[])
    client = MagicMock()
    client.beta.chat.completions.parse.return_value.choices[0].message.parsed = expected

    with patch("src.generate.OpenAI") as openai_cls:
        result = generate_answer("python", docs, settings, client=client)
        openai_cls.assert_not_called()

    assert result is expected
    client.beta.chat.completions.parse.assert_called_once()
