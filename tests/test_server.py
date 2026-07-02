from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from chapterss.server import (
    app,
    fetch_feed,
    list_podcasts,
    podcast_config,
    safe_http_get,
    validate_episode_id,
    validate_podcast_id,
    validate_url,
)


class TestValidatePodcastId:
    """Tests for validate_podcast_id function."""

    def test_valid_podcast_ids(self) -> None:
        """Test that valid podcast IDs are accepted."""
        assert validate_podcast_id("test123") == "test123"
        assert validate_podcast_id("my-podcast") == "my-podcast"
        assert validate_podcast_id("my_podcast") == "my_podcast"
        assert validate_podcast_id("Podcast-123_test") == "Podcast-123_test"

    def test_invalid_podcast_ids(self) -> None:
        """Test that invalid podcast IDs raise HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            validate_podcast_id("../etc/passwd")
        assert exc_info.value.status_code == 400

        with pytest.raises(HTTPException):
            validate_podcast_id("test/podcast")

        with pytest.raises(HTTPException):
            validate_podcast_id("test podcast")

        with pytest.raises(HTTPException):
            validate_podcast_id("test@podcast")


class TestValidateEpisodeId:
    """Tests for validate_episode_id function."""

    def test_valid_episode_ids(self) -> None:
        """Test that valid episode IDs are accepted."""
        assert validate_episode_id("episode123") == "episode123"
        assert validate_episode_id("ep-123") == "ep-123"
        assert validate_episode_id("ep_123") == "ep_123"
        assert validate_episode_id("ep.123") == "ep.123"

    def test_invalid_episode_ids(self) -> None:
        """Test that invalid episode IDs raise HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            validate_episode_id("../etc/passwd")
        assert exc_info.value.status_code == 400

        with pytest.raises(HTTPException):
            validate_episode_id("episode/123")

        with pytest.raises(HTTPException):
            validate_episode_id("episode 123")


class TestValidateUrl:
    """Tests for validate_url function."""

    def test_valid_urls(self) -> None:
        """Test that valid URLs are accepted."""
        assert validate_url("http://example.com") == "http://example.com"
        assert validate_url("https://example.com/feed.xml") == "https://example.com/feed.xml"

    def test_invalid_schemes(self) -> None:
        """Test that non-HTTP(S) schemes are rejected."""
        with pytest.raises(ValueError, match="URL scheme must be http or https"):
            validate_url("ftp://example.com")

        with pytest.raises(ValueError, match="URL scheme must be http or https"):
            validate_url("file:///etc/passwd")

    def test_localhost_blocked(self) -> None:
        """Test that localhost URLs are blocked."""
        with pytest.raises(ValueError, match="Access to localhost is not allowed"):
            validate_url("http://localhost/test")

        with pytest.raises(ValueError, match="Access to localhost is not allowed"):
            validate_url("http://127.0.0.1/test")

        with pytest.raises(ValueError, match="Access to localhost is not allowed"):
            validate_url("http://0.0.0.0/test")

    def test_private_ips_blocked(self) -> None:
        """Test that private IP ranges are blocked."""
        with pytest.raises(ValueError, match="Access to private IP ranges is not allowed"):
            validate_url("http://10.0.0.1/test")

        with pytest.raises(ValueError, match="Access to private IP ranges is not allowed"):
            validate_url("http://192.168.1.1/test")

        with pytest.raises(ValueError, match="Access to private IP ranges is not allowed"):
            validate_url("http://172.16.0.1/test")

    def test_no_hostname(self) -> None:
        """Test that URLs without hostname are rejected."""
        with pytest.raises(ValueError, match="URL must have a valid hostname"):
            validate_url("http://")


class TestSafeHttpGet:
    """Tests for safe_http_get function."""

    def test_safe_http_get_success(self, mocker: MockerFixture, temp_dir: Path) -> None:
        """Test successful HTTP GET with streaming to file."""
        target_path = temp_dir / "output.dat"
        url = "http://example.com/file.dat"

        # Mock response
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_content = MagicMock(return_value=[b"chunk1", b"chunk2", b"chunk3"])

        mock_get = mocker.patch("chapterss.server.requests.get", return_value=mock_response)

        safe_http_get(url, max_size=1000, target_path=target_path)

        # Verify request was made
        mock_get.assert_called_once_with(url, timeout=30, stream=True, headers={"User-Agent": "ChapteRSS/0.2.0"})

        # Verify file was created
        assert target_path.exists()
        content = target_path.read_bytes()
        assert content == b"chunk1chunk2chunk3"

    def test_safe_http_get_creates_parent_directory(self, mocker: MockerFixture, temp_dir: Path) -> None:
        """Test that parent directory is created if it doesn't exist."""
        target_path = temp_dir / "nested" / "dir" / "output.dat"
        url = "http://example.com/file.dat"

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_content = MagicMock(return_value=[b"data"])

        mocker.patch("chapterss.server.requests.get", return_value=mock_response)

        safe_http_get(url, max_size=1000, target_path=target_path)

        assert target_path.parent.exists()
        assert target_path.exists()


class TestListPodcasts:
    """Tests for list_podcasts function."""

    def test_list_podcasts_empty(self, mocker: MockerFixture) -> None:
        """Test listing podcasts when config directory doesn't exist."""
        mocker.patch("chapterss.server.Path.exists", return_value=False)

        result = list_podcasts()

        assert result == {"podcasts": []}

    def test_list_podcasts_with_podcasts(self, mocker: MockerFixture, temp_dir: Path) -> None:
        """Test listing podcasts."""
        config_dir = temp_dir / "config"
        config_dir.mkdir()

        # Create podcast directories
        (config_dir / "podcast1").mkdir()
        (config_dir / "podcast1" / "config.yaml").touch()
        (config_dir / "podcast2").mkdir()
        (config_dir / "podcast2" / "config.yaml").touch()

        # Mock Path.resolve to return our temp config dir
        mocker.patch.object(Path, "resolve", return_value=config_dir)

        result = list_podcasts()

        assert "podcasts" in result
        assert len(result["podcasts"]) == 2
        assert "podcast1" in result["podcasts"]
        assert "podcast2" in result["podcasts"]


class TestPodcastConfig:
    """Tests for podcast_config function."""

    def test_podcast_config_success(self, mocker: MockerFixture, temp_dir: Path) -> None:
        """Test loading valid podcast configuration."""
        config_dir = temp_dir / "config"
        config_dir.mkdir()
        podcast_dir = config_dir / "test-podcast"
        podcast_dir.mkdir()

        config_data = {"source_rss": "https://example.com/feed.xml"}
        config_file = podcast_dir / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        mocker.patch.object(Path, "resolve", return_value=config_dir)

        result = podcast_config("test-podcast")

        assert result == config_data


class TestFetchFeed:
    """Tests for fetch_feed function."""

    def test_fetch_feed_not_exists(self, mocker: MockerFixture, temp_dir: Path) -> None:
        """Test fetching feed when it doesn't exist."""
        data_dir = temp_dir / "data"
        config = {"source_rss": "https://example.com/feed.xml"}

        mocker.patch.object(Path, "resolve", return_value=data_dir)
        mock_safe_get = mocker.patch("chapterss.server.safe_http_get")

        result = fetch_feed("test-podcast", config)

        # Verify feed was fetched
        mock_safe_get.assert_called_once()
        assert result.parent.name == "original"


class TestServerEndpoints:
    """Tests for server endpoints."""

    def test_list_endpoint(self, mocker: MockerFixture, temp_dir: Path) -> None:
        """Test /list endpoint."""
        config_dir = temp_dir / "config"
        config_dir.mkdir()
        (config_dir / "podcast1").mkdir()
        (config_dir / "podcast1" / "config.yaml").touch()

        mocker.patch.object(Path, "resolve", return_value=config_dir)

        client = TestClient(app)
        response = client.get("/list")

        assert response.status_code == 200
        data = response.json()
        assert "podcasts" in data
        assert "podcast1" in data["podcasts"]
