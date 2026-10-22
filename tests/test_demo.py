import hashlib
import json
import re
import shutil
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from scripts import demo
from src.generate import RAGAnswer
from src.retrieve import get_chroma_count, init_chroma, upsert_chroma

REPO = Path(__file__).resolve().parents[1]
OFFLINE_SUMMARY = "Offline extractive summary (set OPENAI_API_KEY for LLM synthesis)."


class HashEmbeddings(Embeddings):
    """Deterministic stand-in so the demo test does not download a model."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
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


def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("PINECONE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def _prepared_root(tmp_path: Path, *, with_resumes: bool = True, env_keys: bool = False) -> Path:
    root = tmp_path / "repo"
    ood = root / "data" / "ood"
    ood.mkdir(parents=True)
    shutil.copy(REPO / "data" / "ood" / "state_of_the_union.txt", ood / "state_of_the_union.txt")
    if with_resumes:
        resumes = root / "data" / "resumes"
        resumes.mkdir()
        for pdf in sorted((REPO / "data" / "resumes").glob("*.pdf")):
            shutil.copy(pdf, resumes / pdf.name)
    if env_keys:
        (root / ".env").write_text(
            "PINECONE_API_KEY=test-pinecone\nOPENAI_API_KEY=test-openai\n",
            encoding="utf-8",
        )
    return root


def _bind_root(monkeypatch: pytest.MonkeyPatch, root: Path) -> dict[str, bool]:
    _isolate_env(monkeypatch)
    monkeypatch.setattr(demo, "ROOT", root)
    called = {"embeddings": False}

    def _embeddings(_settings: object) -> HashEmbeddings:
        called["embeddings"] = True
        return HashEmbeddings()

    monkeypatch.setattr(demo, "get_embeddings", _embeddings)
    return called


def _release_chroma() -> None:
    from chromadb.api.client import SharedSystemClient

    SharedSystemClient.clear_system_cache()


def _answer_from_output(output: str) -> RAGAnswer:
    start = output.index('{\n  "query"')
    payload, _end = json.JSONDecoder().raw_decode(output[start:])
    return RAGAnswer.model_validate(payload)


def test_offline_demo_prints_five_beats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _prepared_root(tmp_path)
    called = _bind_root(monkeypatch, root)

    demo.main()
    output = capsys.readouterr().out

    assert called["embeddings"] is True
    assert "loaded_pages=" in output
    assert "candidates=9" in output
    assert "state_of_the_union.txt was not indexed" in output
    assert "character: chunks=" in output
    assert "recursive: chunks=" in output
    first = output.split("chroma_count_after_first=", 1)[1].splitlines()[0]
    second = output.split("chroma_count_after_second=", 1)[1].splitlines()[0]
    assert first == second
    assert int(first) > 0
    assert re.search(r"(?m)^chroma\s+1\s+\S+", output)
    assert re.search(r"(?m)^faiss\s+1\s+\S+", output)
    assert "Pinecone skipped: PINECONE_API_KEY is not set." in output
    answer = _answer_from_output(output)
    assert answer.query == demo.QUERY
    assert answer.summary == OFFLINE_SUMMARY
    assert answer.matches
    assert (root / ".chroma").is_dir()


def test_missing_corpus_writes_no_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _prepared_root(tmp_path, with_resumes=False)
    _bind_root(monkeypatch, root)

    with pytest.raises(FileNotFoundError, match="Resume directory not found"):
        demo.main()

    assert not (root / ".chroma").exists()


def test_second_run_resets_existing_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _prepared_root(tmp_path)
    _bind_root(monkeypatch, root)
    settings = demo.load_settings()
    chroma_db = init_chroma(settings, HashEmbeddings(), persist_dir=settings.chroma_persist_dir)
    upsert_chroma(
        chroma_db,
        [Document(page_content="seed row", metadata={"candidate_id": "seed"})],
        ["seed-id"],
    )
    assert get_chroma_count(chroma_db) == 1
    _release_chroma()

    demo.main()
    output = capsys.readouterr().out
    first = int(output.split("chroma_count_after_first=", 1)[1].splitlines()[0])
    second = int(output.split("chroma_count_after_second=", 1)[1].splitlines()[0])
    assert first == second
    assert first > 1

    _release_chroma()
    reopened = init_chroma(settings, HashEmbeddings(), persist_dir=settings.chroma_persist_dir)
    stored = reopened._collection.get(include=[])
    assert "seed-id" not in stored["ids"]
    assert get_chroma_count(reopened) == second


def test_keys_present_includes_pinecone_and_model_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _prepared_root(tmp_path, env_keys=True)
    _bind_root(monkeypatch, root)
    settings = demo.load_settings()
    assert settings.has_pinecone is True
    assert settings.has_openai is True

    def _pinecone(chunks, ids, embeddings, query, bound_settings, k=3):
        assert bound_settings.has_pinecone is True
        assert query == demo.QUERY
        assert k == 3
        assert chunks and len(chunks) == len(ids)
        assert all("::recursive::" in doc_id for doc_id in ids)
        assert embeddings is not None
        return [
            {
                "id": ids[0],
                "score": 0.91,
                "metadata": {"candidate_id": "01_alex_rivera"},
            }
        ]

    def _generate(query, retrieved_docs, bound_settings, client=None):
        assert bound_settings.has_openai is True
        assert query == demo.QUERY
        assert retrieved_docs
        assert client is None
        return RAGAnswer(
            query=query,
            summary="Model wrote this synthesis.",
            matches=[],
        )

    monkeypatch.setattr(demo, "upsert_and_query_pinecone", _pinecone)
    monkeypatch.setattr(demo, "generate_answer", _generate)

    demo.main()
    output = capsys.readouterr().out
    assert re.search(r"(?m)^pinecone\s+1\s+01_alex_rivera\s+0\.91$", output)
    assert "Pinecone skipped" not in output
    answer = _answer_from_output(output)
    assert answer.summary == "Model wrote this synthesis."
    assert answer.query == demo.QUERY


def test_configured_store_failure_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _prepared_root(tmp_path, env_keys=True)
    _bind_root(monkeypatch, root)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("pinecone down")

    monkeypatch.setattr(demo, "upsert_and_query_pinecone", _boom)

    with pytest.raises(RuntimeError, match="pinecone down"):
        demo.main()


def test_configured_answer_failure_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _prepared_root(tmp_path, env_keys=True)
    _bind_root(monkeypatch, root)
    monkeypatch.setattr(demo, "upsert_and_query_pinecone", lambda *_args, **_kwargs: [])

    def _boom(*_args, **_kwargs):
        raise RuntimeError("openai down")

    monkeypatch.setattr(demo, "generate_answer", _boom)

    with pytest.raises(RuntimeError, match="openai down"):
        demo.main()
