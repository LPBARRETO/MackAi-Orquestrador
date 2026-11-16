# MackAi-Orquestrador (Módulo 7)

Microsserviço responsável pela orquestração de chamadas ao LLM (`google-genai` / `gemini-2.5-flash`), construção dinâmica de prompts por perfil (`Desenvolvedor`, `Product Owner`, `Gerente de Projetos`), integração com o serviço interno de busca semântica (`mackai-busca`) e tratamento resiliente de falhas (`mode=fallback` e `mode=offline`).

## Contratos da API (Porta 8000 no container / 8001 local)

- `GET /health`: Sonda de vida/prontidão do Container Apps (`{"status": "ok"}`).
- `GET /v1/status`: Retorna o estado do LLM (`gemini` ou `offline`), modelo ativo e estatísticas do `mackai-busca`.
- `POST /v1/answers`: Recebe `question`, `profile`, `temperature`, `history` e `project_id` (opcional), consulta o `mackai-busca` (`/v1/search`) repassando o `project_id` e gera a resposta contextualizada.

## Como rodar localmente

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # Adicione sua GEMINI_API_KEY (opcional)
uvicorn orquestracao_llm.main:app --reload --port 8001