import argparse
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

import feedparser
import requests
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from feedgen.feed import FeedGenerator

from .insert_chapters import process_episode

log: logging.Logger = logging.getLogger(__name__)

app: FastAPI = FastAPI()

# Security constants
MAX_FEED_SIZE: int = 10 * 1024 * 1024  # 10MB
MAX_AUDIO_SIZE: int = 500 * 1024 * 1024  # 500MB
REQUEST_TIMEOUT: int = 30  # seconds
ALLOWED_SCHEMES: set[str] = {"http", "https"}


def validate_podcast_id(podcast_id: str) -> str:
    """Validate and sanitize podcast_id to prevent path traversal."""
    # Only allow alphanumeric, underscore, and hyphen
    if not re.match(r"^[a-zA-Z0-9_-]+$", podcast_id):
        raise HTTPException(status_code=400, detail="Invalid podcast ID format")
    return podcast_id


def validate_episode_id(episode_id: str) -> str:
    """Validate and sanitize episode_id to prevent path traversal."""
    # Only allow alphanumeric, underscore, hyphen, and dots (but not ..)
    if not re.match(r"^[a-zA-Z0-9_.-]+$", episode_id) or ".." in episode_id:
        raise HTTPException(status_code=400, detail="Invalid episode ID format")
    return episode_id


def validate_url(url: str) -> str:
    """Validate URL to prevent SSRF attacks."""
    parsed = urlparse(url)

    # Check scheme
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"URL scheme must be http or https, got: {parsed.scheme}")

    # Prevent access to local/private networks
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL must have a valid hostname")

    # Block localhost and private IP ranges
    if hostname.lower() in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        raise ValueError("Access to localhost is not allowed")

    # Block private IP ranges (basic check)
    if hostname.startswith(
        (
            "10.",
            "172.16.",
            "172.17.",
            "172.18.",
            "172.19.",
            "172.20.",
            "172.21.",
            "172.22.",
            "172.23.",
            "172.24.",
            "172.25.",
            "172.26.",
            "172.27.",
            "172.28.",
            "172.29.",
            "172.30.",
            "172.31.",
            "192.168.",
            "169.254.",
        )
    ):
        raise ValueError("Access to private IP ranges is not allowed")

    return url


def safe_http_get(url: str, max_size: int, target_path: Path, timeout: int = REQUEST_TIMEOUT) -> None:
    validate_url(url)

    try:
        response = requests.get(url, timeout=timeout, stream=True, headers={"User-Agent": "ChapteRSS/0.2.0"})
        response.raise_for_status()

        # Stream to temporary file, then rename
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Create temp file in same directory as target for atomic rename
        fd, temp_path = tempfile.mkstemp(dir=target_path.parent, suffix=".tmp")
        try:
            total_size = 0
            with os.fdopen(fd, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        total_size += len(chunk)
                        if total_size > max_size:
                            raise ValueError(f"Response too large (max {max_size} bytes)")
                        f.write(chunk)

            # Atomic rename
            os.rename(temp_path, target_path)
        except Exception:
            # Clean up temp file on error
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    except requests.RequestException as e:
        log.error(f"HTTP request failed for {url}: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to fetch resource: {str(e)}")


@app.get("/list")
def list_podcasts() -> Dict[str, List[str]]:
    """List all available podcasts by scanning for config.yaml files."""
    config_dir: Path = Path("config").resolve()
    if not config_dir.exists():
        return {"podcasts": []}

    podcasts: List[str] = []
    for config_file in config_dir.glob("*/config.yaml"):
        podcast_id: str = config_file.parent.name
        # Validate the podcast_id
        if re.match(r"^[a-zA-Z0-9_-]+$", podcast_id):
            podcasts.append(podcast_id)

    return {"podcasts": podcasts}


def podcast_config(podcast_id: str) -> Dict[str, Any]:
    """Load and validate podcast configuration."""
    podcast_id = validate_podcast_id(podcast_id)

    config_path: Path = Path("config").resolve() / podcast_id / "config.yaml"

    # Ensure the path is within the config directory
    try:
        config_path = config_path.resolve()
        config_path.relative_to(Path("config").resolve())
    except ValueError, RuntimeError:
        raise HTTPException(status_code=400, detail="Invalid podcast configuration path")

    if not config_path.exists():
        raise HTTPException(status_code=404, detail="Podcast configuration not found")

    try:
        with open(config_path, "r") as f:
            config: Any = yaml.safe_load(f)
    except yaml.YAMLError as e:
        log.error(f"Failed to parse config for {podcast_id}: {e}")
        raise HTTPException(status_code=500, detail="Invalid podcast configuration")

    # Validate required fields
    if not isinstance(config, dict) or "source_rss" not in config:
        raise HTTPException(status_code=500, detail="Invalid podcast configuration: missing source_rss")

    # Validate the source_rss URL
    try:
        validate_url(config["source_rss"])
    except ValueError as e:
        log.error(f"Invalid source_rss URL for {podcast_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Invalid source RSS URL: {str(e)}")

    return config


def fetch_feed(podcast_id: str, config: Dict[str, Any]) -> Path:
    """
    Fetch and cache the original RSS feed.
    Returns the path to the cached feed file.
    Refreshes the feed if it doesn't exist or is older than 1 hour.
    """
    data_dir: Path = Path("data").resolve()
    original: Path = data_dir / podcast_id / "original" / "feed.xml"
    original.parent.mkdir(parents=True, exist_ok=True)

    # Refresh original feed if it doesn't exist or is older than 1 hour
    should_refresh_original = False
    if not original.exists():
        should_refresh_original = True
    else:
        age_seconds = time.time() - original.stat().st_mtime
        if age_seconds > 3600:  # 1 hour
            should_refresh_original = True

    if should_refresh_original:
        safe_http_get(config["source_rss"], MAX_FEED_SIZE, original)

    return original


@app.get("/rss/{podcast_id}")
def rss(podcast_id: str, request: Request) -> FileResponse:
    """Serve RSS feed with modified enclosure URLs."""
    podcast_id = validate_podcast_id(podcast_id)
    config: Dict[str, Any] = podcast_config(podcast_id)

    data_dir: Path = Path("data").resolve()

    # Fetch or use cached feed
    original: Path = fetch_feed(podcast_id, config)

    chapped: Path = data_dir / podcast_id / "chapped" / "feed.xml"
    chapped.parent.mkdir(parents=True, exist_ok=True)

    # Refresh chapped feed if it doesn't exist or is older than original
    should_refresh_chapped = False
    if not chapped.exists():
        should_refresh_chapped = True
    elif original.exists() and chapped.stat().st_mtime < original.stat().st_mtime:
        should_refresh_chapped = True

    if should_refresh_chapped:
        parsed_feed: feedparser.FeedParserDict = feedparser.parse(str(original))

        scheme: str = request.headers.get("x-forwarded-proto", request.url.scheme)
        host: str = request.headers.get("host", request.url.netloc)
        base_url: str = f"{scheme}://{host}"

        # Create new feed with feedgen
        fg: FeedGenerator = FeedGenerator()
        fg.title(parsed_feed.feed.get("title", "Podcast") + " (With Chapters)")
        fg.link(href=parsed_feed.feed.get("link", ""), rel="alternate")
        fg.description(parsed_feed.feed.get("description", ""))
        fg.generator("ChapteRSS - https://github.com/shish/chapterss")

        # Copy over other feed-level attributes if they exist
        if "language" in parsed_feed.feed:
            fg.language(parsed_feed.feed.language)
        if "image" in parsed_feed.feed:
            fg.image(parsed_feed.feed.image.get("href", ""))

        # Add entries with modified enclosure URLs
        for entry in parsed_feed.entries:
            fe = fg.add_entry()
            entry_id: str = entry.get("id", entry.get("link", ""))
            # Sanitize entry ID for use in URL
            safe_entry_id: str = re.sub(r"[^a-zA-Z0-9_.-]", "_", entry_id)

            fe.id(entry_id)
            fe.title(entry.get("title", ""))
            fe.link(href=entry.get("link", ""))

            if "description" in entry:
                fe.description(entry.description)
            if "published" in entry:
                fe.published(entry.published)
            if "author" in entry:
                fe.author(name=entry.author)

            # Modify enclosure URL to point to our server
            if entry.get("enclosures"):
                original_enclosure = entry.enclosures[0]
                url: str = f"{base_url}/audio/{podcast_id}/{safe_entry_id}.mp3"
                fe.enclosure(
                    url=url,
                    length=original_enclosure.get("length", "0"),
                    type=original_enclosure.get("type", "audio/mpeg"),
                )

        chapped.write_text(fg.rss_str(pretty=True).decode("utf-8"))

    return FileResponse(chapped)


@app.get("/audio/{podcast_id}/{episode_id}.mp3")
def audio(podcast_id: str, episode_id: str) -> FileResponse:
    """Serve audio file with chapters."""
    podcast_id = validate_podcast_id(podcast_id)
    episode_id = validate_episode_id(episode_id)
    config: Dict[str, Any] = podcast_config(podcast_id)

    data_dir: Path = Path("data").resolve()
    config_dir: Path = Path("config").resolve()

    original: Path = data_dir / podcast_id / "original" / f"{episode_id}.mp3"
    original.parent.mkdir(parents=True, exist_ok=True)

    if not original.exists():
        # Fetch or use cached feed
        cached_feed: Path = fetch_feed(podcast_id, config)
        try:
            feed_content: bytes = cached_feed.read_bytes()
            feed: feedparser.FeedParserDict = feedparser.parse(feed_content)
        except Exception as e:
            log.error(f"Failed to read cached RSS feed: {e}")
            raise HTTPException(status_code=500, detail="Failed to read cached RSS feed")

        for entry in feed.entries:
            # Try to match both the original ID and sanitized ID
            entry_id: str = entry.get("id", entry.get("link", ""))
            safe_entry_id: str = re.sub(r"[^a-zA-Z0-9_.-]", "_", entry_id)

            if entry_id == episode_id or safe_entry_id == episode_id:
                if entry.get("enclosures"):
                    url: str = entry.enclosures[0].href
                    try:
                        safe_http_get(url, MAX_AUDIO_SIZE, target_path=original)
                        break
                    except Exception as e:
                        log.error(f"Failed to fetch audio: {e}")
                        raise HTTPException(status_code=502, detail="Failed to fetch audio file")
        else:
            raise HTTPException(status_code=404, detail="Episode not found")

    chapped: Path = data_dir / podcast_id / "chapped" / f"{episode_id}.mp3"
    chapped.parent.mkdir(parents=True, exist_ok=True)

    if not chapped.exists():
        markers_dir: Path = config_dir / podcast_id / "markers"
        process_episode(
            original,
            markers_dir,
            chapped,
            transcribe=True,
            summarise=True,
        )

    return FileResponse(chapped)


def main() -> None:
    """Run the uvicorn server with the FastAPI app."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Run the ChapteRSS podcast server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", default=False, help="Auto-reload on code changes")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    args: argparse.Namespace = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("chapterss").setLevel(log_level)

    uvicorn.run(
        "chapterss.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
