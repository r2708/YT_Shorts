"""LLM_PROVIDER=claude-cli: the pipeline's model calls go through `claude -p`."""
import json
import subprocess

import pytest
from pydantic import BaseModel
from typing import List

import claude_cli_backend
import llm_backend


class Item(BaseModel):
    id: int
    score: int


class Answer(BaseModel):
    items: List[Item]


def _fake_run(envelope, seen):
    def run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["prompt"] = kwargs.get("input")
        seen["cwd"] = kwargs.get("cwd")
        import os
        seen["files"] = sorted(os.listdir(kwargs["cwd"]))
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(envelope), stderr="")
    return run


@pytest.fixture
def cli(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "claude-cli")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setattr(claude_cli_backend, "available", lambda: True)


def test_provider_routing(cli):
    assert llm_backend.provider() == "claude-cli"
    assert llm_backend.active()
    assert llm_backend.describe()["provider"] == "claude-cli"
    assert "Claude CLI" in llm_backend.label()


def test_inactive_by_default(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    assert not claude_cli_backend.active()
    assert not llm_backend.active()


def test_structured_output_is_validated(cli, monkeypatch):
    seen = {}
    env = {"is_error": False, "structured_output": {"items": [{"id": 1, "score": 80}]},
           "usage": {"input_tokens": 5, "output_tokens": 7}, "total_cost_usd": 0.01}
    monkeypatch.setattr(subprocess, "run", _fake_run(env, seen))
    parsed, cost = llm_backend.generate_json("score these", Answer)
    assert parsed == {"items": [{"id": 1, "score": 80}]}
    assert cost["total_cost"] == 0.0 and cost["local"] is True
    assert "-p" in seen["cmd"] and "--json-schema" in seen["cmd"]
    schema = json.loads(seen["cmd"][seen["cmd"].index("--json-schema") + 1])
    assert "$ref" not in json.dumps(schema)  # refs are inlined
    assert seen["cmd"][seen["cmd"].index("--tools") + 1] == ""  # text call: no tools


def test_falls_back_to_result_text(cli, monkeypatch):
    env = {"is_error": False, "result": '```json\n{"items": [{"id": 2, "score": 9}]}\n```'}
    monkeypatch.setattr(subprocess, "run", _fake_run(env, {}))
    parsed, _ = claude_cli_backend.generate_json("x", Answer)
    assert parsed["items"][0]["id"] == 2


def test_images_are_written_and_read_tool_enabled(cli, monkeypatch):
    seen = {}
    env = {"is_error": False, "structured_output": {"items": []}}
    monkeypatch.setattr(subprocess, "run", _fake_run(env, seen))
    claude_cli_backend.generate_json("look", Answer, images=[b"a", ("Shot 1", b"b")])
    assert seen["files"] == ["img_000.jpg", "img_001.jpg"]
    assert "Shot 1: ./img_001.jpg" in seen["prompt"]
    assert seen["cmd"][seen["cmd"].index("--tools") + 1] == "Read"


def test_errors_raise(cli, monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run(
        {"is_error": True, "result": "rate limited", "api_error_status": 429}, {}))
    with pytest.raises(RuntimeError, match="Claude CLI"):
        claude_cli_backend.generate_json("x", Answer)
    monkeypatch.setattr(subprocess, "run", _fake_run(
        {"is_error": False, "structured_output": {"items": [{"id": "nope"}]}}, {}))
    with pytest.raises(Exception):
        claude_cli_backend.generate_json("x", Answer)
