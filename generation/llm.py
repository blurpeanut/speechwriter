"""
GPT-4o streaming call via OpenAI SDK.
Only permitted LLM provider: api.openai.com.
"""
import json
import logging
import os
from typing import Generator

from openai import OpenAI

logger = logging.getLogger(__name__)


def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set. Add it to your .env file.")
    return OpenAI(api_key=api_key)


def stream_speech(
    system_prompt: str,
    user_message: str,
    model: str | None = None,
) -> Generator[str, None, None]:
    """
    Stream GPT-4o response tokens.
    Yields string chunks as they arrive from the API.
    The caller is responsible for accumulating the full response.
    """
    model = model or os.getenv("LLM_MODEL", "gpt-4o")
    client = _client()

    logger.info("Streaming from %s …", model)
    with client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.7,
        stream=True,
    ) as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


def generate_speech(
    system_prompt: str,
    user_message: str,
    model: str | None = None,
) -> dict:
    """
    Non-streaming wrapper: accumulates the full streamed response and parses
    the JSON returned by GPT-4o.

    Returns the parsed dict with keys: outline, full_speech, style_confidence_score.
    Raises ValueError if the response is not valid JSON.
    """
    accumulated = ""
    for token in stream_speech(system_prompt, user_message, model=model):
        accumulated += token

    # Strip markdown code fences if the model wraps output in ```json … ```
    text = accumulated.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("LLM response is not valid JSON: %s", exc)
        logger.debug("Raw response (first 500 chars): %s", accumulated[:500])
        raise ValueError(f"LLM did not return valid JSON: {exc}") from exc

    return result
