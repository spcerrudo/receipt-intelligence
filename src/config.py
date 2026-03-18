from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(slots=True)
class AppConfig:
    """Central application configuration."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_root: Path = field(default_factory=lambda: Path(os.getenv("DATA_ROOT", "data/raw")))
    output_root: Path = field(default_factory=lambda: Path(os.getenv("OUTPUT_ROOT", "outputs")))
    ocr_lang: str = os.getenv("OCR_LANG", "en")
    use_preprocessing: bool = _to_bool(os.getenv("USE_PREPROCESSING"), True)
    min_field_confidence: float = float(os.getenv("MIN_FIELD_CONFIDENCE", "0.65"))
    llm_provider: str = os.getenv("LLM_PROVIDER", "bedrock").strip().lower()
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL") or None
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    bedrock_model_id: str = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0")
    bedrock_mm_model_id: str = os.getenv("BEDROCK_MM_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0")

    def __post_init__(self) -> None:
        self.data_root = self._resolve(self.data_root)
        self.output_root = self._resolve(self.output_root)
        self.ensure_directories()

    def _resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else self.project_root / path

    def ensure_directories(self) -> None:
        for path in [
            self.output_root,
            self.output_root / "raw_ocr",
            self.output_root / "corrected_entities",
            self.output_root / "metrics",
            self.output_root / "figures",
            self.project_root / "data" / "interim",
            self.project_root / "data" / "processed",
        ]:
            path.mkdir(parents=True, exist_ok=True)


def get_config() -> AppConfig:
    """Return a populated configuration object."""

    return AppConfig()
