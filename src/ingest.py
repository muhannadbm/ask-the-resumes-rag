from pathlib import Path
from typing import Literal
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

def load_pdf_resumes(directory: Path) -> list[Document]:
    """Loads only .pdf files in deterministic sorted order, ignoring .txt or other files."""
    if not directory.exists():
        raise FileNotFoundError(f"Resume directory not found: {directory}")

    pdf_files = sorted(directory.glob("*.pdf"))
    docs: list[Document] = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        for page_doc in loader.load():
            page_doc.metadata["candidate_id"] = pdf_path.stem
            page_doc.metadata["source_file"] = pdf_path.name
            docs.append(page_doc)
    return docs

def split_documents(
    docs: list[Document],
    method: Literal["character", "recursive"] = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> tuple[list[Document], list[str], list[int]]:
    """
    Splits documents, prepends [Candidate: ID] context, and returns:
      (enriched_chunks, deterministic_ids, token_length_counts)
    """
    if method == "character":
        splitter = CharacterTextSplitter(
            separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            separators=[
                r"\n(?=Professional summary\. |Selected work on |Experience\. |Education\. )",
                r"(?<![A-Z]\.[A-Z])\.\s+(?=[A-Z])",
                "\n",
                " ",
                "",
            ],
            is_separator_regex=True,
            chunk_size=560,
            chunk_overlap=80,
        )

    raw_chunks = splitter.split_documents(docs)
    enriched_chunks: list[Document] = []
    deterministic_ids: list[str] = []
    token_counts: list[int] = []

    per_candidate_counter: dict[str, int] = {}

    for chunk in raw_chunks:
        cand_id = chunk.metadata.get("candidate_id", "unknown")
        idx = per_candidate_counter.get(cand_id, 0)
        per_candidate_counter[cand_id] = idx + 1

        chunk_id = f"{cand_id}::{method}::chunk_{idx}"
        header = f"[Candidate: {cand_id}]\n"
        text = (
            chunk.page_content
            if chunk.page_content.startswith("[Candidate:")
            else f"{header}{chunk.page_content}"
        )

        new_meta = {
            **chunk.metadata,
            "chunk_id": chunk_id,
            "chunk_index": idx,
            "splitter": method,
        }
        enriched_chunks.append(Document(page_content=text, metadata=new_meta))
        deterministic_ids.append(chunk_id)
        token_counts.append(len(text.split()))

    return enriched_chunks, deterministic_ids, token_counts
