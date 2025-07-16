# test_example.py

import pytest
from typer.testing import CliRunner
import openai
import example

runner = CliRunner()


class DummyQuotaError(openai.error.RateLimitError):
    pass


class DummyOtherError(openai.error.OpenAIError):
    pass


@pytest.fixture(autouse=True)
def patch_openai(monkeypatch):
    calls = {}

    def fake_create(*args, **kwargs):
        if kwargs.get("mock"):
            return "MOCKED"
        mode = calls.get("mode")
        if mode == "quota":
            raise DummyQuotaError("insufficient_quota")
        if mode == "other":
            raise DummyOtherError("something went wrong")
        class Choice:
            message = type("M", (), {"content":
                "A unicorn snoozed under the stars."
            })
        return type("R", (), {"choices": [Choice()]})

    monkeypatch.setattr(
        openai.ChatCompletion, "create", lambda *a, **k: fake_create(**k)
    )
    return calls


def test_quota_exceeded_message(patch_openai):
    patch_openai["mode"] = "quota"
    result = runner.invoke(example.app, ["--mock=False"])
    assert result.exit_code == 2
    assert (
        "❌ Quota exceeded – please check your plan at "
        "https://platform.openai.com/account/billing/plan"
        in result.stdout
    )


def test_other_api_error(patch_openai):
    patch_openai["mode"] = "other"
    result = runner.invoke(example.app, [])
    assert result.exit_code == 3
    assert "API ERROR ▶" in result.stdout


def test_successful_response(patch_openai):
    result = runner.invoke(example.app, ["--mock=False"])
    assert result.exit_code == 0
    assert "A unicorn snoozed under the stars." in result.stdout


def test_mock_mode():
    result = runner.invoke(example.app, ["--mock"])
    assert result.exit_code == 0
    assert (
        "Once upon a time, in a land of code, "
        "there was a sleeping unicorn."
        in result.stdout
    )
