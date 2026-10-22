from langchain_core.documents import Document
from openai import OpenAI
from pydantic import BaseModel, Field
from src.config import Settings

class CandidateMatch(BaseModel):
    candidate_id: str = Field(description="Candidate identifier from [Candidate: ID]")
    matching_skills: list[str] = Field(description="Skills from the resume matching the query")
    evidence_quote: str = Field(description="Direct quote from the resume chunk")

class RAGAnswer(BaseModel):
    query: str
    summary: str = Field(description="Concise synthesis answering the recruiter query")
    matches: list[CandidateMatch] = Field(default_factory=list)

SYSTEM_PROMPT = (
    "You are a technical recruiting assistant. Answer the query ONLY using the provided "
    "resume chunks. Cite exact quotes. If the query is unrelated to resumes or no candidate "
    "matches, return an empty matches list and state that no matching candidates were found."
)

def generate_answer(
    query: str,
    retrieved_docs: list[Document],
    settings: Settings,
    client: OpenAI | None = None,
) -> RAGAnswer:
    context_block = "\n\n---\n\n".join(
        f"Chunk ID: {d.metadata.get('chunk_id')}\n{d.page_content}" for d in retrieved_docs
    )
    if client is None and not settings.has_openai:
        # Offline fallback when OPENAI_API_KEY is not configured
        matches = [
            CandidateMatch(
                candidate_id=str(d.metadata.get("candidate_id", "unknown")),
                matching_skills=[],
                evidence_quote=d.page_content[:200].strip(),
            )
            for d in retrieved_docs
        ]
        return RAGAnswer(
            query=query,
            summary="Offline extractive summary (set OPENAI_API_KEY for LLM synthesis).",
            matches=matches,
        )

    llm_client = client or OpenAI(api_key=settings.openai_api_key.get_secret_value())
    completion = llm_client.beta.chat.completions.parse(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {query}\n\nResume Chunks:\n{context_block}"},
        ],
        response_format=RAGAnswer,
        temperature=0.0,
    )
    return completion.choices[0].message.parsed
