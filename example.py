# example.py

import asyncio
from typing import Optional

import openai
import typer
from loguru import logger
from pydantic import BaseSettings, ValidationError
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)
from prometheus_client import Counter, Histogram, start_http_server


class Settings(BaseSettings):
    openai_api_key: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


try:
    settings = Settings()
    openai.api_key = settings.openai_api_key
except ValidationError as e:
    logger.error("Configuration error: {error}", error=e)
    raise typer.Exit(code=1)


REQUEST_COUNT = Counter(
    "bedtime_request_total", "Total number of bedtime story requests"
)
REQUEST_LATENCY = Histogram(
    "bedtime_request_latency_seconds",
    "Latency for bedtime story API calls",
)
ERROR_COUNT = Counter(
    "bedtime_errors_total",
    "Total number of errors during bedtime story requests",
)

start_http_server(8000)


@retry(
    retry=retry_if_exception_type(
        (openai.error.RateLimitError, openai.error.Timeout)
    ),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _call_openai_sync(prompt: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        timeout=10,
    )
    return response.choices[0].message.content.strip()


async def generate_story(mock: bool = False) -> str:
    REQUEST_COUNT.inc()
    with REQUEST_LATENCY.time():
        if mock:
            logger.info("Mock mode enabled – returning dummy story")
            return (
                "Once upon a time, in a land of code, "
                "there was a sleeping unicorn."
            )
        try:
            story = await asyncio.to_thread(
                _call_openai_sync,
                "Write a one-sentence bedtime story about a unicorn.",
            )
            return story
        except openai.error.RateLimitError:
            ERROR_COUNT.inc()
            typer.secho(
                "❌ Quota exceeded – please check your plan at "
                "https://platform.openai.com/account/billing/plan",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=2)
        except openai.error.OpenAIError as e:
            ERROR_COUNT.inc()
            typer.secho(
                f"API ERROR ▶ {e}",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=3)


app = typer.Typer(
    help="Generate a one-sentence bedtime story about a unicorn."
)


@app.command()
def main(
    mock: Optional[bool] = typer.Option(
        False, "--mock", help="Run in offline mock mode"
    )
):
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="{time} | {level} | {message}",
        level="INFO",
        serialize=True,
    )
    story = asyncio.run(generate_story(mock))
    typer.secho(story, fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
