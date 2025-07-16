# example.py

import sys
import asyncio
import signal
from enum import IntEnum
from typing import Optional

import openai
import typer
from loguru import logger
from pydantic import BaseSettings, Field, ValidationError
from prometheus_client import (
    Counter,
    Histogram,
    start_http_server,
    make_asgi_app,
)
from starlette.applications import Starlette
from starlette.routing import Mount

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# ─── 1) CONFIG ────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    model: str = Field("gpt-3.5-turbo", env="OPENAI_MODEL")
    temperature: float = Field(0.7, env="OPENAI_TEMPERATURE")
    timeout: int = Field(10, env="OPENAI_TIMEOUT")
    metrics_port: int = Field(8000, env="METRICS_PORT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


try:
    settings = Settings()
    openai.api_key = settings.openai_api_key
except ValidationError as e:
    logger.error("Configuration error: {error}", error=e)
    sys.exit(ExitCode.CONFIG_ERROR)

# ─── 2) METRICS ───────────────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "bedtime_request_total", "Total number of bedtime story requests"
)
REQUEST_LATENCY = Histogram(
    "bedtime_request_latency_seconds", "Latency for bedtime story API calls"
)
ERROR_COUNT = Counter(
    "bedtime_errors_total", "Total number of errors during bedtime story requests"
)

# Spin up an ASGI app for Prometheus on /metrics
app_metrics = Starlette(
    routes=[
        Mount("/metrics", make_asgi_app(), name="metrics")
    ],
)

# Start the HTTP server in its own thread
start_http_server(settings.metrics_port)

# ─── 3) EXIT CODES ────────────────────────────────────────────────────────────

class ExitCode(IntEnum):
    SUCCESS = 0
    CONFIG_ERROR = 1
    QUOTA_EXCEEDED = 2
    API_ERROR = 3

# ─── 4) CLI ───────────────────────────────────────────────────────────────────

app = typer.Typer(help="Generate a one-sentence bedtime story about a unicorn.")
app.add_typer(typer.Typer(name="version"), name="version")


@app.command("version")
def version():
    """Show application version."""
    typer.echo("example.py v1.0.0")

# ─── 5) HELPER ────────────────────────────────────────────────────────────────

def _call_openai_sync(
    prompt: str,
    model: str,
    temperature: float,
    timeout: int,
) -> str:
    """Blocking call to OpenAI; returns the story text."""
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        timeout=timeout,
    )
    return resp.choices[0].message.content.strip()


# Retry on rate-limit or timeout, up to 3 times with exponential backoff
@retry(
    retry=retry_if_exception_type(
        (openai.error.RateLimitError, openai.error.Timeout)
    ),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
)
def call_openai_with_retry(prompt: str) -> str:
    return _call_openai_sync(
        prompt,
        settings.model,
        settings.temperature,
        settings.timeout,
    )

# ─── 6) BUSINESS LOGIC ───────────────────────────────────────────────────────

async def generate_story(mock: bool) -> str:
    """
    Generate or mock a one-sentence unicorn bedtime story.

    Exits with:
      SUCCESS (0) – prints the story
      CONFIG_ERROR (1) – missing API key
      QUOTA_EXCEEDED (2) – rate limit or insufficient quota
      API_ERROR (3) – other OpenAI errors
    """
    REQUEST_COUNT.inc()
    with REQUEST_LATENCY.time():
        if mock:
            logger.info("Mock mode enabled – returning dummy story")
            return "Once upon a time, in a land of code, there was a sleeping unicorn."

        try:
            # Offload blocking call to thread
            return await asyncio.to_thread(
                call_openai_with_retry,
                "Write a one-sentence bedtime story about a unicorn.",
            )
        except openai.error.RateLimitError as e:
            ERROR_COUNT.inc()
            typer.secho(
                "❌ Quota exceeded – please check your plan at "
                "https://platform.openai.com/account/billing/plan",
                fg=typer.colors.RED,
            )
            sys.exit(ExitCode.QUOTA_EXCEEDED)
        except openai.error.OpenAIError as e:
            ERROR_COUNT.inc()
            typer.secho(f"API ERROR ▶ {e}", fg=typer.colors.RED)
            sys.exit(ExitCode.API_ERROR)


# ─── 7) ENTRYPOINT ───────────────────────────────────────────────────────────


@app.command()
def main(
    mock: bool = typer.Option(
        False,
        "--mock",
        help="Use a canned story without calling the API",
    )
):
    """CLI entrypoint."""
    # Handle CTRL+C gracefully
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: sys.exit(ExitCode.SUCCESS))

    story = asyncio.run(generate_story(mock))
    typer.secho(story, fg=typer.colors.GREEN)
    sys.exit(ExitCode.SUCCESS)


if __name__ == "__main__":
    app()    
