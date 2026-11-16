from fastapi import APIRouter, Header, Response
from orquestracao_llm.modelos import AnswerRequest, AnswerResponse, StatusResponse
from orquestracao_llm.pipeline import answer_question, get_orchestrator_status

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/v1/status", response_model=StatusResponse)
async def status(
    response: Response,
    x_request_id: str | None = Header(default=None, alias="X-Request-ID"),
) -> StatusResponse:
    if x_request_id:
        response.headers["X-Request-ID"] = x_request_id
    return await get_orchestrator_status(request_id=x_request_id)


@router.post("/v1/answers", response_model=AnswerResponse)
async def answers(
    payload: AnswerRequest,
    response: Response,
    x_request_id: str | None = Header(default=None, alias="X-Request-ID"),
) -> AnswerResponse:
    if x_request_id:
        response.headers["X-Request-ID"] = x_request_id
    return await answer_question(payload, request_id=x_request_id)