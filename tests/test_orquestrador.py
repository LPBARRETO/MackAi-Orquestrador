import uuid
from fastapi.testclient import TestClient
import pytest
from orquestracao_llm.config import settings
from orquestracao_llm.main import app
from orquestracao_llm.modelos import BuscaStats, TrechoBusca, UsageOut
from orquestracao_llm import pipeline

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_perfil_invalido_retorna_422():
    r = client.post(
        "/v1/answers",
        json={"question": "Olá", "profile": "Estagiario", "temperature": 0.4},
    )
    assert r.status_code == 422


def test_modo_offline_sem_chave(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "")

    async def fake_search(query, project_id=None, request_id=None):
        return [
            TrechoBusca(
                source="modulo-7",
                path="docs/README.md",
                start_line=1,
                end_line=20,
                similarity=0.85,
                text="Conteúdo do documento indexado",
            )
        ]

    monkeypatch.setattr(pipeline.busca_client, "search", fake_search)

    r = client.post(
        "/v1/answers",
        json={"question": "Como funciona o projeto?", "profile": "Desenvolvedor"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "offline"
    assert body["grounded"] is True
    assert len(body["sources"]) == 1
    assert body["usage"] is None


def test_gemini_sucesso_e_repasse_project_id(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "fake-key")
    captured = {}

    async def fake_search(query, project_id=None, request_id=None):
        captured["query"] = query
        captured["project_id"] = str(project_id) if project_id else None
        return [
            TrechoBusca(
                source="modulo-7",
                path="services/orquestrador/main.py",
                start_line=10,
                end_line=40,
                similarity=0.91,
                text="Código de exemplo do orquestrador",
            )
        ]

    async def fake_generate(system_instruction, prompt, temperature, request_id=None):
        captured["system"] = system_instruction
        captured["prompt"] = prompt
        return "Resposta gerada pelo Gemini", UsageOut(input_tokens=120, output_tokens=45)

    monkeypatch.setattr(pipeline.busca_client, "search", fake_search)
    monkeypatch.setattr(pipeline.gemini_client, "generate", fake_generate)

    pid = str(uuid.uuid4())
    r = client.post(
        "/v1/answers",
        json={
            "question": "E para o PO?",
            "profile": "Product Owner",
            "temperature": 0.3,
            "project_id": pid,
            "campo_extra_futuro": "ignorado_sem_erro",
            "history": [
                {"role": "user", "content": "Quais User Stories podemos simplificar?"},
                {"role": "assistant", "content": "As histórias 64 e 89."},
            ],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "gemini"
    assert body["grounded"] is True
    assert body["usage"] == {"input_tokens": 120, "output_tokens": 45}
    # Verifica reescrita de pergunta curta com histórico e repasse de project_id
    assert "Quais User Stories podemos simplificar?" in captured["query"]
    assert captured["project_id"] == pid
    assert "<trechos>" in captured["prompt"]
    assert "<pergunta>" in captured["prompt"]


def test_gemini_erro_retorna_fallback_200(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "fake-key")

    async def fake_search(query, project_id=None, request_id=None):
        return [
            TrechoBusca(
                source="modulo-7",
                path="arq.py",
                start_line=1,
                end_line=10,
                similarity=0.75,
                text="Trecho salvo mesmo no fallback",
            )
        ]

    async def fake_generate_fail(system_instruction, prompt, temperature, request_id=None):
        return None

    monkeypatch.setattr(pipeline.busca_client, "search", fake_search)
    monkeypatch.setattr(pipeline.gemini_client, "generate", fake_generate_fail)

    r = client.post(
        "/v1/answers",
        json={"question": "Explique o código", "profile": "Desenvolvedor"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "fallback"
    assert body["grounded"] is True
    assert len(body["sources"]) == 1
    assert body["usage"] is None


def test_busca_fora_do_ar_responde_grounded_false(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "fake-key")

    async def fake_search_empty(query, project_id=None, request_id=None):
        return []

    async def fake_generate(system_instruction, prompt, temperature, request_id=None):
        return "Resposta com conhecimento geral", UsageOut(input_tokens=50, output_tokens=30)

    monkeypatch.setattr(pipeline.busca_client, "search", fake_search_empty)
    monkeypatch.setattr(pipeline.gemini_client, "generate", fake_generate)

    r = client.post(
        "/v1/answers",
        json={"question": "O que é Chaos Engineering?", "profile": "Desenvolvedor"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "gemini"
    assert body["grounded"] is False
    assert body["sources"] == []


def test_status_endpoint(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "fake-key")

    async def fake_stats(request_id=None):
        return BuscaStats(strategy="embedding", documents=42, chunks=380, indexed_at="2026-11-12T10:00:00Z")

    monkeypatch.setattr(pipeline.busca_client, "stats", fake_stats)

    r = client.get("/v1/status")
    assert r.status_code == 200
    body = r.json()
    assert body["llm"] == "gemini"
    assert body["busca"]["documents"] == 42