import argparse
import logging
import os

from anthropic import Anthropic
from dotenv import load_dotenv

log: logging.Logger = logging.getLogger(__name__)


def summarise(text: str, max_words: int = 10, api_key: str | None = None) -> str:
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    if max_words < 1:
        raise ValueError(f"max_words must be at least 1, got {max_words}")

    # Get API key from parameter or environment
    if api_key is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError(
            "Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter"
        )

    log.debug(f"Summarising text: {text[:100]}...")

    client = Anthropic(api_key=api_key)

    try:
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

        summary = response.content[0].text.strip()
        log.debug(f"Summary: {summary}")
        return summary
    except Exception as e:
        log.error(f"Anthropic API error: {e}")
        raise RuntimeError(f"Failed to summarise text: {e}")


def main() -> None:
    load_dotenv()

    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Summarise text using Anthropic Claude API")
    parser.add_argument("text", type=str, help="Text to summarise")
    parser.add_argument(
        "--max-words", type=int, default=10, help="Maximum number of words in the summary (default: 10)"
    )
    parser.add_argument(
        "--api-key", type=str, help="Anthropic API key (default: reads from ANTHROPIC_API_KEY environment variable)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    args: argparse.Namespace = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("chapterss").setLevel(log_level)

    try:
        summary: str = summarise(args.text, max_words=args.max_words, api_key=args.api_key)
        print(summary)
    except Exception as e:
        log.error(f"Error: {e}")
        exit(1)
