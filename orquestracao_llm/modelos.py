from typing import Literal
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field

ProfileType = Literal["Desenvolvedor", "Product Owner", "Gerente de Projetos"]
ModeType = Literal["gemini", "offline", "fallback"]


class HistoryItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class AnswerRequest(BaseModel):
    # extra="ignore" garante que qualquer campo adicional enviado pelo Conversas nunca quebre com 422/503
    model_config = ConfigDict(extra="ignore")

    question: str = Field(min_length=1, max_length=20000)
    profile: ProfileType = "Desenvolvedor"
    temperature: float = Field(default=0.4, ge=0.0, le=1.0)
    history: list[HistoryItem] = Field(default_factory=list, max_length=10)
    project_id: UUID | str | None = None


class TrechoBusca(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source: str = "modulo-7"
    path: str
    start_line: int
    end_line: int
    similarity: float
    text: str


class BuscaSearchResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    strategy: str = "embedding"
    results: list[TrechoBusca] = Field(default_factory=list)


class SourceOut(BaseModel):
    source: str
    path: str
    start_line: int
    end_line: int
    similarity: float
    excerpt: str


class UsageOut(BaseModel):
    input_tokens: int
    output_tokens: int


class AnswerResponse(BaseModel):
    content: str
    mode: ModeType
    grounded: bool
    sources: list[SourceOut]
    model: str
    usage: UsageOut | None = None


class BuscaStats(BaseModel):
    model_config = ConfigDict(extra="ignore")

    strategy: str = "embedding"
    documents: int = 0
    chunks: int = 0
    indexed_at: str | None = None


class StatusResponse(BaseModel):
    llm: Literal["gemini", "offline"]
    model: str
    busca: BuscaStats | None = None