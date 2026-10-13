from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


def _load_llm(monkeypatch, host="https://e2-dogfood.staging.cloud.databricks.com/"):
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
    databricks_sdk.WorkspaceClient = lambda *, profile: SimpleNamespace(
        profile=profile, config=SimpleNamespace(host=host)
    )
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


def test_wrong_profile_host_fails_before_calling_the_model(monkeypatch) -> None:
    llm, captured = _load_llm(monkeypatch, host="https://dogfood.staging.databricks.com")

    with pytest.raises(RuntimeError, match="points at https://dogfood.staging.databricks.com"):
        llm.complete("system", "user")
    assert "messages" not in captured


def test_saved_setup_profile_is_reused_on_plain_rerun(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("BRIEF_AGENT_PROFILE", raising=False)
    llm, captured = _load_llm(monkeypatch)
    profile_path = tmp_path / ".brief-agent-profile"
    profile_path.write_text("brief-agent\n")
    monkeypatch.setattr(llm, "_PROFILE_PATH", profile_path)
    assert llm.complete("system", "user") == "brief"
    assert captured["workspace_client"].profile == "brief-agent"


def test_explicit_profile_overrides_saved_selection(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("BRIEF_AGENT_PROFILE", "another-valid-profile")
    llm, captured = _load_llm(monkeypatch)
    profile_path = tmp_path / ".brief-agent-profile"
    profile_path.write_text("brief-agent\n")
    monkeypatch.setattr(llm, "_PROFILE_PATH", profile_path)
    llm.complete("system", "user")
    assert captured["workspace_client"].profile == "another-valid-profile"


def test_fresh_clone_keeps_original_dogfood_default(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("BRIEF_AGENT_PROFILE", raising=False)
    llm, captured = _load_llm(monkeypatch)
    monkeypatch.setattr(llm, "_PROFILE_PATH", tmp_path / "absent")
    llm.complete("system", "user")
    assert captured["workspace_client"].profile == "dogfood"
