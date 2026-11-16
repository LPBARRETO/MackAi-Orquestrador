import logging
import time
from orquestracao_llm.cliente_busca import busca_client
from orquestracao_llm.cliente_gemini import gemini_client
from orquestracao_llm.config import settings
from orquestracao_llm.modelos import (
    AnswerRequest,
    AnswerResponse,
    HistoryItem,
    SourceOut,
    StatusResponse,
    TrechoBusca,
)
from orquestracao_llm.prompts import (
    FALLBACK_MESSAGE,
    build_offline_answer,
    build_system_instruction,
    build_user_prompt,
)

logger = logging.getLogger("mackai.orquestrador.pipeline")


def build_search_query(question: str, history: list[HistoryItem]) -> str:
    palavras = question.strip().split()
    if len(palavras) < 8 and history:
        ultimas_perguntas = [h.content for h in reversed(history) if h.role == "user"]
        if ultimas_perguntas:
            combinada = f"{ultimas_perguntas[0].strip()} {question.strip()}"
            return combinada[:4000]
    return question.strip()[:4000]


def map_sources(trechos: list[TrechoBusca]) -> list[SourceOut]:
    return [
        SourceOut(
            source=t.source,
            path=t.path,
            start_line=t.start_line,
            end_line=t.end_line,
            similarity=round(float(t.similarity), 2),
            excerpt=t.text[:300],
        )
        for t in trechos
    ]


async def answer_question(
    req: AnswerRequest,
    request_id: str | None = None,
) -> AnswerResponse:
    inicio = time.perf_counter()
    historico_limitado = req.history[-settings.history_max_messages :]

    # 1. Monta a consulta de busca (reescrita em perguntas curtas de acompanhamento)
    query_busca = build_search_query(req.question, historico_limitado)

    # 2. Chama a Busca repassando project_id (quando informado pelo Conversas)
    trechos = await busca_client.search(
        query=query_busca,
        project_id=req.project_id,
        request_id=request_id,
    )

    # 3. Grounded é verdadeiro quando a Busca devolveu pelo menos um trecho
    grounded = len(trechos) > 0
    sources_out = map_sources(trechos)

    # 4. Modo offline se não houver GEMINI_API_KEY configurada
    if not settings.is_gemini_enabled:
        conteudo = build_offline_answer(req.question, req.profile, trechos)
        latencia_ms = int((time.perf_counter() - inicio) * 1000)
        logger.info(
            "request_id=%s mode=offline grounded=%s latency_ms=%d input_tokens=0 output_tokens=0",
            request_id or "-",
            grounded,
            latencia_ms,
        )
        return AnswerResponse(
            content=conteudo,
            mode="offline",
            grounded=grounded,
            sources=sources_out,
            model=settings.gemini_model,
            usage=None,
        )

    # 5. Monta o prompt e chama o Gemini
    system_instruction = build_system_instruction(req.profile)
    user_prompt = build_user_prompt(req.question, trechos, historico_limitado)

    resultado_llm = await gemini_client.generate(
        system_instruction=system_instruction,
        prompt=user_prompt,
        temperature=req.temperature,
        request_id=request_id,
    )

    latencia_ms = int((time.perf_counter() - inicio) * 1000)

    # 6. Tratamento de falha do LLM -> mode=fallback (US 80) mantendo status 200 e as fontes
    if resultado_llm is None:
        logger.warning(
            "request_id=%s mode=fallback grounded=%s latency_ms=%d input_tokens=0 output_tokens=0",
            request_id or "-",
            grounded,
            latencia_ms,
        )
        return AnswerResponse(
            content=FALLBACK_MESSAGE,
            mode="fallback",
            grounded=grounded,
            sources=sources_out,
            model=settings.gemini_model,
            usage=None,
        )

    texto, usage = resultado_llm
    in_tokens = usage.input_tokens if usage else 0
    out_tokens = usage.output_tokens if usage else 0

    logger.info(
        "request_id=%s mode=gemini grounded=%s latency_ms=%d input_tokens=%d output_tokens=%d",
        request_id or "-",
        grounded,
        latencia_ms,
        in_tokens,
        out_tokens,
    )

    return AnswerResponse(
        content=texto,
        mode="gemini",
        grounded=grounded,
        sources=sources_out,
        model=settings.gemini_model,
        usage=usage,
    )


async def get_orchestrator_status(request_id: str | None = None) -> StatusResponse:
    stats_busca = await busca_client.stats(request_id=request_id)
    llm_state = "gemini" if settings.is_gemini_enabled else "offline"
    return StatusResponse(
        llm=llm_state,
        model=settings.gemini_model,
        busca=stats_busca,
    )