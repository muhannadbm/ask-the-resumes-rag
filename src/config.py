from pathlib import Path
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    resumes_dir: Path = Path("data/resumes")
    ood_dir: Path = Path("data/ood")
    chroma_persist_dir: Path = Path(".chroma")
    chroma_collection_name: str = "resumes_collection"

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    chunk_size: int = 1000
    chunk_overlap: int = 100

    # Optional external API keys (never hardcoded)
    pinecone_api_key: SecretStr | None = None
    pinecone_index_name: str = "resume-rag-384"
    openai_api_key: SecretStr | None = None
    llm_model: str = "gpt-4o-mini"

    @property
    def has_pinecone(self) -> bool:
        return bool(self.pinecone_api_key and self.pinecone_api_key.get_secret_value().strip())

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key.get_secret_value().strip())
