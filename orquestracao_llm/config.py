import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "").strip()
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
    gemini_timeout_ms: int = int(os.getenv("GEMINI_TIMEOUT_MS", "30000"))
    max_output_tokens: int = int(os.getenv("MAX_OUTPUT_TOKENS", "2048"))
    busca_url: str = os.getenv("BUSCA_URL", "http://localhost:8002").rstrip("/")
    busca_timeout_s: float = float(os.getenv("BUSCA_TIMEOUT_S", "5"))
    busca_top_k: int = int(os.getenv("BUSCA_TOP_K", "3"))
    history_max_messages: int = int(os.getenv("HISTORY_MAX_MESSAGES", "10"))

    @property
    def is_gemini_enabled(self) -> bool:
        return bool(self.gemini_api_key)


settings = Settings()