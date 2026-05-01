from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from bbsia.core.config import settings

DEFAULT_MODEL = settings.ollama_generation.default_model
DEFAULT_TOP_K = settings.retrieval.top_k


class ChatRequest(BaseModel):
    pergunta: str
    modelo: str = DEFAULT_MODEL
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)
    filtro_area: list[str] = Field(default_factory=list)
    filtro_assunto: list[str] = Field(default_factory=list)
    conversation_id: str | None = None

    @field_validator("pergunta")
    @classmethod
    def validar_pergunta(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("O campo 'pergunta' e obrigatorio e nao pode estar vazio.")
        return value.strip()


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)
    filtro_area: list[str] = Field(default_factory=list)
    filtro_assunto: list[str] = Field(default_factory=list)

    @field_validator("query")
    @classmethod
    def validar_query(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("O campo 'query' e obrigatorio e nao pode estar vazio.")
        return value.strip()


class ChunkResult(BaseModel):
    score: float
    id: int
    documento: str
    pagina: int | None
    area: str
    assuntos: list[str]
    doc_titulo: str = ""
    doc_autores: list[str] = Field(default_factory=list)
    doc_ano: int | None = None
    section_heading: str | None = None
    content_type: str = "text"
    parent_id: str | None = None
    ocr_usado: bool = False
    table_index: int | None = None
    texto: str
    chunk_index: int | None


class ChatResponse(BaseModel):
    resposta: str
    fontes: list[str]
    resultados: list[ChunkResult]
    modelo_usado: str
    total_fontes: int
    total_chunks_recuperados: int


class SearchResponse(BaseModel):
    query: str
    total: int
    resultados: list[ChunkResult]


class UploadMetadataRequest(BaseModel):
    documento: str
    area: str = Field(min_length=1)
    assuntos: list[str] = Field(default_factory=list)

    @field_validator("documento", "area")
    @classmethod
    def validar_campos_texto(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Campo obrigatorio nao pode estar vazio.")
        return value.strip()

    @field_validator("assuntos")
    @classmethod
    def validar_assuntos(cls, value: list[str]) -> list[str]:
        cleaned = [v.strip() for v in value if isinstance(v, str) and v.strip()]
        return cleaned or ["geral"]
