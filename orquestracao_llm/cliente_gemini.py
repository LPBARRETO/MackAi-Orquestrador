import asyncio
import logging
from google import genai
from google.genai import types
from orquestracao_llm.config import settings
from orquestracao_llm.modelos import UsageOut

logger = logging.getLogger("mackai.orquestrador.gemini")


class GeminiClient:
    def __init__(self) -> None:
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client | None:
        if not settings.is_gemini_enabled:
            return None
        if self._client is None:
            self._client = genai.Client(
                api_key=settings.gemini_api_key,
                http_options=types.HttpOptions(timeout=settings.gemini_timeout_ms),
            )
        return self._client

    async def generate(
        self,
        system_instruction: str,
        prompt: str,
        temperature: float,
        request_id: str | None = None,
    ) -> tuple[str, UsageOut | None] | None:
        client = self._get_client()
        if client is None:
            return None

        for attempt in (1, 2):
            try:
                resposta = await client.aio.models.generate_content(
                    model=settings.gemini_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=temperature,
                        max_output_tokens=settings.max_output_tokens,
                    ),
                )
                texto = (resposta.text or "").strip()
                if not texto:
                    logger.warning(
                        "Gemini retornou texto vazio (request_id=%s, tentativa=%d)",
                        request_id or "-",
                        attempt,
                    )
                    return None

                uso_meta = getattr(resposta, "usage_metadata", None)
                usage = None
                if uso_meta is not None:
                    in_tok = getattr(uso_meta, "prompt_token_count", 0) or 0
                    out_tok = getattr(uso_meta, "candidates_token_count", 0) or 0
                    usage = UsageOut(input_tokens=in_tok, output_tokens=out_tok)

                return texto, usage

            except Exception as exc:
                status_str = str(exc)
                is_transient = "429" in status_str or "503" in status_str
                logger.warning(
                    "Erro na chamada ao Gemini (request_id=%s, tentativa=%d): %s",
                    request_id or "-",
                    attempt,
                    exc,
                )
                if attempt == 1 and is_transient:
                    await asyncio.sleep(1.5)
                    continue
                return None

        return None


gemini_client = GeminiClient()