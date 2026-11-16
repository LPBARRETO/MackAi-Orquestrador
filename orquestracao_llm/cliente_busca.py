import logging
from uuid import UUID
import httpx
from orquestracao_llm.config import settings
from orquestracao_llm.modelos import BuscaSearchResponse, BuscaStats, TrechoBusca

logger = logging.getLogger("mackai.orquestrador.busca")


class BuscaClient:
    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=settings.busca_url,
                timeout=httpx.Timeout(settings.busca_timeout_s),
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def search(
        self,
        query: str,
        project_id: UUID | str | None = None,
        request_id: str | None = None,
    ) -> list[TrechoBusca]:
        payload: dict = {
            "query": query[:4000],
            "top_k": settings.busca_top_k,
        }
        if project_id is not None:
            payload["project_id"] = str(project_id)

        headers: dict[str, str] = {}
        if request_id:
            headers["X-Request-ID"] = request_id

        try:
            client = self._get_client()
            resp = await client.post("/v1/search", json=payload, headers=headers)
            resp.raise_for_status()
            parsed = BuscaSearchResponse.model_validate(resp.json())
            return parsed.results
        except Exception as exc:
            logger.warning(
                "Falha ao consultar mackai-busca (request_id=%s): %s",
                request_id or "-",
                exc,
            )
            return []

    async def stats(self, request_id: str | None = None) -> BuscaStats | None:
        headers: dict[str, str] = {}
        if request_id:
            headers["X-Request-ID"] = request_id

        try:
            client = self._get_client()
            resp = await client.get("/v1/stats", headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return BuscaStats.model_validate(data)
        except Exception as exc:
            logger.warning(
                "Falha ao obter /v1/stats de mackai-busca (request_id=%s): %s",
                request_id or "-",
                exc,
            )
            return None


busca_client = BuscaClient()