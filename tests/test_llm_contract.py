from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace


def _load_llm(monkeypatch):
    captured: dict[str, object] = {}

    class FakeChatDatabricks:
        def __init__(self, *, endpoint, workspace_client):
            captured["endpoint"] = endpoint
            captured["workspace_client"] = workspace_client

        def invoke(self, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return SimpleNamespace(content="brief")

    databricks_sdk = ModuleType("databricks.sdk")
    databricks_sdk.WorkspaceClient = lambda *, profile: {"profile": profile}
    databricks_langchain = ModuleType("databricks_langchain")
    databricks_langchain.ChatDatabricks = FakeChatDatabricks
    langchain_messages = ModuleType("langchain_core.messages")
    langchain_messages.HumanMessage = lambda content: ("human", content)
    langchain_messages.SystemMessage = lambda content: ("system", content)

    monkeypatch.setitem(sys.modules, "databricks.sdk", databricks_sdk)
    monkeypatch.setitem(sys.modules, "databricks_langchain", databricks_langchain)
    monkeypatch.setitem(sys.modules, "langchain_core.messages", langchain_messages)
    sys.modules.pop("plain.brief_agent.llm", None)
    return importlib.import_module("plain.brief_agent.llm"), captured


def test_brief_generation_uses_terra_compatible_request(monkeypatch) -> None:
    llm, captured = _load_llm(monkeypatch)

    assert llm.complete("system", "user") == "brief"
    assert captured["endpoint"] == "databricks-gpt-5-6-terra"
    assert captured["kwargs"] == {
        "max_tokens": 12_000,
        "reasoning_effort": "low",
    }
