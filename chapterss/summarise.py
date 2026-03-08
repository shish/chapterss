import argparse
import logging
import os

from anthropic import Anthropic
from anthropic.types import TextBlock

log: logging.Logger = logging.getLogger(__name__)


def summarise(text: str, max_words: int = 10, api_key: str | None = None) -> str:
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    if max_words < 1:
        raise ValueError(f"max_words must be at least 1, got {max_words}")
    if api_key is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter"
        )

    log.debug(f"Summarising text: {text[:100]}...")
    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=100,
        temperature=0.7,
        messages=[
            {
                "role": "user",
                "content": f"Summarise the following text in {max_words} words or less. Provide only the summary, no other commentary:\n\n{text}",
            }
        ],
    )

    block = response.content[0]
    if not isinstance(block, TextBlock):
        raise ValueError(f"Unexpected response format from Anthropic API: {response.content}")
    summary = block.text.strip()
    log.debug(f"Summary: {summary}")
    return summary


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Summarise text using Anthropic Claude API")
    parser.add_argument("text", type=str, help="Text to summarise")
    parser.add_argument("--max-words", type=int, default=10, help="Maximum number of words in the summary")
    parser.add_argument("--api-key", type=str, help="Anthropic API key")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    args: argparse.Namespace = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("chapterss").setLevel(log_level)

    summary: str = summarise(args.text, max_words=args.max_words, api_key=args.api_key)
    print(summary)
