"""JSON utility functions."""

import json
import re
from typing import Any

from ..utils.exceptions import LLMException
from ..utils.logger import logger


def clean_json_response(response: str) -> dict[str, Any]:
    """
    Clean and parse JSON response from LLM.

    Handles common issues like:
    - JSON wrapped in markdown code blocks
    - Extra whitespace
    - Invalid JSON formatting

    Args:
        response: Raw response string from LLM

    Returns:
        Parsed JSON dictionary

    Raises:
        LLMException: If JSON parsing fails
    """
    if not response:
        raise LLMException("Empty response received")

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    try:
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
        if match:
            json_content = match.group(1).strip()
            return json.loads(json_content)
    except json.JSONDecodeError as e:
        logger.error(
            "Failed to parse JSON from markdown block",
            extra={
                "error": str(e),
                "response_preview": response[:200] + "..." if len(response) > 200 else response,
            },
        )

    try:
        cleaned = re.sub(r",\s*}", "}", response)
        cleaned = re.sub(r",\s*]", "]", cleaned)

        json_match = re.search(r"[{\[].*[}\]]", cleaned, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
    except json.JSONDecodeError as e:
        logger.error(
            "Failed to parse JSON after cleaning attempts",
            extra={
                "error": str(e),
                "original_response": response[:500] + "..." if len(response) > 500 else response,
            },
        )

    raise LLMException(
        "Failed to parse JSON response after all attempts",
        details={"response": response[:1000] + "..." if len(response) > 1000 else response},
    )


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """
    Safely serialize object to JSON string.

    Args:
        obj: Object to serialize
        **kwargs: Additional arguments for json.dumps

    Returns:
        JSON string
    """
    try:
        return json.dumps(obj, **kwargs)
    except (TypeError, ValueError) as e:
        logger.error(
            "Failed to serialize object to JSON",
            extra={"error": str(e), "object_type": type(obj).__name__},
        )
        return json.dumps(str(obj), **kwargs)
