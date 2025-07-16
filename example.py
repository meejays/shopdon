# example.py

import os
import sys
import asyncio

import openai
import typer
from loguru import logger

app = typer.Typer()

# Default dummy story for mock mode
DEFAULT_DUMMY = "Once upon a time, in a land of code, there was a sleeping unicorn."

def _call_openai_sync(prompt: str) -> str:
    """Blocking call to OpenAI ChatCompletion; returns the story text."""
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip()

async def generate_story(mock: bool) -> str:
    """
    Generate or mock a one-sentence unicorn bedtime story.
    Exits with:
      1 if API key is missing,
      2 on quota errors,
      3 on other OpenAI errors.
    """
    if mock:
        logger.info("Mock mode enabled")
        return DEFAULT_DUMMY

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        typer.echo("❌ OPENAI_API_KEY not set")
        sys.exit(1)

    openai.api_key = api_key
    prompt = "Write a one-sentence bedtime story about a unicorn."

    try:
        # Run blocking OpenAI call in thread to avoid blocking the event loop
        story = await asyncio.get_event_loop().run_in_executor(
            None, _call_openai_sync, prompt
        )
        return story
    except Exception as e:
        # Quota / rate-limit errors
        if "insufficient_quota" in str(e) or isinstance(e, openai.error.RateLimitError):
            typer.secho("❌ Quota exceeded – please check your plan at https://platform.openai.com/account/billing/plan", fg=typer.colors.RED)

            sys.exit(2)
        # Other OpenAI errors
        typer.echo(f"API ERROR ▶ {e}")
        sys.exit(3)

@app.command()
def main(mock: bool = typer.Option(False, "--mock", help="Use a canned story without calling the API")):
    """CLI entrypoint."""
    story = asyncio.run(generate_story(mock))
    typer.echo(story)

if __name__ == "__main__":
    app()
