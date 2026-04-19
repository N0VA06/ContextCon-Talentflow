"""
AWS Bedrock LLM Client
Wraps Claude via Bedrock with robust JSON parsing and error handling.

Key fix: _parse_json_robust handles:
  - Markdown fences (```json ... ```)
  - Truncated JSON (hits max_tokens limit) — extracts largest valid object
  - Trailing garbage after valid JSON
  - Raw text fallback with {"raw_text": ...}
"""

import asyncio
import json
import logging
import re
import time
from typing import Any, Optional, List

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from config import settings

load_dotenv()
logger = logging.getLogger(__name__)

ANTHROPIC_VERSION = "bedrock-2023-05-31"

# Default max tokens — high enough for complex JSON responses
# Claude Haiku 4.5 supports 8192 output tokens
DEFAULT_MAX_TOKENS = 8192


class BedrockLLM:
    """Synchronous + async Bedrock Claude wrapper."""

    def __init__(self):
        if not settings.AWS_ACCESS_KEY_ID or not settings.AWS_SECRET_ACCESS_KEY:
            raise ValueError(
                "AWS credentials missing. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env"
            )
        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )
        self.profile = settings.CLAUDE_INFERENCE_PROFILE
        logger.info("✅ Bedrock client ready (region=%s)", settings.AWS_REGION)

    # ── sync ───────────────────────────────────────────────────────────────────

    def invoke(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        t0 = time.monotonic()
        # Use higher default to avoid truncation of complex JSON
        effective_max = max_tokens or settings.MAX_TOKENS or DEFAULT_MAX_TOKENS
        body: dict = {
            "anthropic_version": ANTHROPIC_VERSION,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": effective_max,
            "temperature": temperature if temperature is not None else settings.TEMPERATURE,
        }
        if system:
            body["system"] = system
        if stop_sequences:
            body["stop_sequences"] = stop_sequences

        try:
            resp = self.client.invoke_model(
                modelId=self.profile,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
            data = json.loads(resp["body"].read())
            text = data["content"][0]["text"].strip()
            stop_reason = data.get("stop_reason", "")
            elapsed = time.monotonic() - t0
            logger.info("Bedrock invoke %.2fs — %d chars (stop=%s)", elapsed, len(text), stop_reason)
            if stop_reason == "max_tokens":
                logger.warning(
                    "⚠️ Response was TRUNCATED at max_tokens=%d — consider increasing MAX_TOKENS",
                    effective_max,
                )
            return text
        except ClientError as e:
            code = e.response["Error"]["Code"]
            msg  = e.response["Error"]["Message"]
            logger.error("Bedrock ClientError %s: %s", code, msg)
            raise
        except Exception as e:
            logger.error("Bedrock error: %s", e)
            raise

    def invoke_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """Invoke and parse response as JSON with robust repair."""
        raw = self.invoke(prompt, system=system, max_tokens=max_tokens)
        return _parse_json_robust(raw)

    # ── async ──────────────────────────────────────────────────────────────────

    async def ainvoke(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        loop = asyncio.get_event_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.invoke(
                        prompt,
                        system=system,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stop_sequences=stop_sequences,
                    ),
                ),
                timeout=settings.BEDROCK_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"LLM timeout after {settings.BEDROCK_TIMEOUT}s")

    async def ainvoke_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> dict:
        """Async invoke + robust JSON parsing."""
        raw = await self.ainvoke(
            prompt, system=system, max_tokens=max_tokens,
            temperature=temperature, stop_sequences=stop_sequences,
        )
        return _parse_json_robust(raw)

    def __call__(self, prompt: str) -> str:
        return self.invoke(prompt)


# ── Robust JSON parser ─────────────────────────────────────────────────────────

def _parse_json_robust(raw: str) -> dict:
    """
    Parse LLM output as JSON with multiple repair strategies:
    1. Strip markdown fences (```json ... ```)
    2. Try direct json.loads
    3. Extract JSON using brace-counting (handles truncation + trailing garbage)
    4. Try to repair truncated JSON by closing open braces/brackets
    5. Return {"raw_text": raw} as absolute fallback
    """
    if not raw:
        return {}

    # Step 1: Strip markdown fences
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first line (```json or ```) and last line (```) if present
        start = 1
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        cleaned = "\n".join(lines[start:end]).strip()

    # Step 2: Direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Step 3: Find JSON object by brace counting
    extracted = _extract_json_object(cleaned)
    if extracted:
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass

    # Step 4: Repair truncated JSON
    repaired = _repair_truncated_json(cleaned)
    if repaired and repaired != cleaned:
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    # Step 5: Last resort — try to find ANY valid JSON in the text
    # Look for any {...} block
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.warning("_parse_json_robust: all strategies failed, returning raw_text (%d chars)", len(raw))
    return {"raw_text": raw}


def _extract_json_object(text: str) -> Optional[str]:
    """
    Extract the outermost JSON object using brace counting.
    Handles extra text before/after the JSON.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        c = text[i]
        if escape_next:
            escape_next = False
            continue
        if c == "\\" and in_string:
            escape_next = True
            continue
        if c == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None


def _repair_truncated_json(text: str) -> str:
    """
    Attempt to repair truncated JSON by closing unclosed braces/brackets.
    Works best for LLM output cut off by max_tokens.
    """
    # Find start of outermost object
    start = text.find("{")
    if start == -1:
        return text

    fragment = text[start:]

    # Count unclosed structures
    depth_brace  = 0
    depth_bracket = 0
    in_string = False
    escape_next = False
    last_valid_pos = 0

    for i, c in enumerate(fragment):
        if escape_next:
            escape_next = False
            continue
        if c == "\\" and in_string:
            escape_next = True
            continue
        if c == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth_brace += 1
        elif c == "}":
            depth_brace -= 1
            if depth_brace == 0:
                last_valid_pos = i
        elif c == "[":
            depth_bracket += 1
        elif c == "]":
            depth_bracket -= 1

    if depth_brace <= 0 and depth_bracket <= 0:
        # Nothing to repair
        return fragment

    # Trim at last complete structure boundary to avoid partial strings
    # Then close open brackets
    # Simple approach: remove incomplete last value and close structures
    trimmed = fragment.rstrip()
    # Remove trailing incomplete token (comma, colon, partial string, partial number)
    trimmed = re.sub(r',\s*$', '', trimmed)
    trimmed = re.sub(r':\s*$', ': null', trimmed)
    trimmed = re.sub(r'"[^"]*$', '"..."', trimmed)  # close partial string

    # Close open brackets
    closing = ("]" * max(depth_bracket, 0)) + ("}" * max(depth_brace, 0))
    return trimmed + closing


# ── singleton ──────────────────────────────────────────────────────────────────

_llm: Optional[BedrockLLM] = None


def get_llm() -> BedrockLLM:
    global _llm
    if _llm is None:
        _llm = BedrockLLM()
    return _llm
