from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from orquestracao_llm.cliente_busca import busca_client
from orquestracao_llm.rotas import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await busca_client.close()


app = FastAPI(
    title="MackAI - Microsserviço Orquestrador",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": str(exc.detail), "code": f"HTTP_{exc.status_code}"},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logging.getLogger("mackai.orquestrador").exception("Erro inesperado: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Erro interno no serviço orquestrador.", "code": "INTERNAL_ERROR"},
    )


app.include_router(router)