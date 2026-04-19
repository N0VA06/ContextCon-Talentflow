"""
TalentFlow Configuration
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # ── CrustData ──────────────────────────────────────────────────────────────
    CRUSTDATA_API_KEY: str = os.getenv("CRUSTDATA_API_KEY", "")
    CRUSTDATA_BASE_URL: str = "https://api.crustdata.com"
    CRUSTDATA_API_VERSION: str = "2025-11-01"

    # ── AWS Bedrock ────────────────────────────────────────────────────────────
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    CLAUDE_INFERENCE_PROFILE: str = os.getenv(
        "CLAUDE_INFERENCE_PROFILE",
        "arn:aws:bedrock:us-east-1:149536491531:inference-profile/us.anthropic.claude-haiku-4-5-20251001-v1:0",
    )
    BEDROCK_TIMEOUT: int = int(os.getenv("BEDROCK_TIMEOUT", "120"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "8192"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.1"))

    # ── App ────────────────────────────────────────────────────────────────────
    APP_TITLE: str = "TalentFlow API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))


settings = Settings()
