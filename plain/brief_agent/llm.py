"""Model access for the brief generator."""
import json
import logging
import os

from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

PROFILE = os.environ.get("BRIEF_AGENT_PROFILE", "dogfood")
MODEL_ENDPOINT = "databricks-gpt-5-6-terra"
MAX_OUTPUT_TOKENS = 12_000
MIN_RETRY_OUTPUT_TOKENS = 12_000

_model = None


def _get_model() -> ChatDatabricks:
    global _model
    if _model is None:
        _model = ChatDatabricks(
            endpoint=MODEL_ENDPOINT,
            workspace_client=WorkspaceClient(profile=PROFILE),
        )
    return _model


def extract_text(content) -> str:
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except (TypeError, ValueError):
                return stripped
            if isinstance(parsed, list) and any(
                isinstance(block, dict) and block.get("type") in {"reasoning", "text"}
                for block in parsed
            ):
                content = parsed
            else:
                return stripped
        else:
            return stripped
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text") or "")
            elif isinstance(block, str):
                texts.append(block)
        if texts:
            return "\n".join(t for t in texts if t).strip()
        logger.warning("Model returned no visible answer")
        return ""
    return str(content or "").strip()


def complete(system: str, user: str, max_tokens: int = MAX_OUTPUT_TOKENS,
             reasoning_effort: str = "low") -> str:
    messages = []
    if system:
        messages.append(SystemMessage(system))
    messages.append(HumanMessage(user))

    model = _get_model()
    response = model.invoke(
        messages,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
    )
    answer = extract_text(response.content)
    if answer:
        return answer

    retry_tokens = max(max_tokens * 2, MIN_RETRY_OUTPUT_TOKENS)
    logger.warning("Empty model response, retrying once")
    response = model.invoke(
        messages,
        max_tokens=retry_tokens,
        reasoning_effort=reasoning_effort,
    )
    return extract_text(response.content)
