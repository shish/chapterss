"""Tests for insert_chapters module."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from chapterss.detect_markers import Chapter
from chapterss.insert_chapters import (
    embed_chapters,
    process_episode,
    write_ffmpeg_metadata,
)


class TestWriteFFmpegMetadata:
    """Tests for write_ffmpeg_metadata function."""

    def test_write_ffmpeg_metadata_basic(self, temp_dir: Path) -> None:
        """Test writing FFmpeg metadata file."""
        chapters = [
            Chapter(start=0.0, end=100.0, title="Intro"),
            Chapter(start=100.0, end=200.0, title="Chapter 1"),
        ]
        output_path = temp_dir / "metadata.txt"

        write_ffmpeg_metadata(chapters, output_path)

        assert output_path.exists()
        content = output_path.read_text()

        assert ";FFMETADATA1" in content
        assert "[CHAPTER]" in content
        assert "title=Intro" in content
        assert "title=Chapter 1" in content
        assert "START=0" in content
        assert "END=100000" in content

    def test_write_ffmpeg_metadata_timing(self, temp_dir: Path) -> None:
        """Test that timing is converted to milliseconds correctly."""
        chapters = [
            Chapter(start=1.5, end=5.75, title="Test"),
        ]
        output_path = temp_dir / "metadata.txt"

        write_ffmpeg_metadata(chapters, output_path)

        content = output_path.read_text()
        assert "START=1500" in content
        assert "END=5750" in content

    def test_write_ffmpeg_metadata_multiple_chapters(self, temp_dir: Path) -> None:
        """Test writing metadata with multiple chapters."""
        chapters = [
            Chapter(start=0.0, end=10.0, title="One"),
            Chapter(start=10.0, end=20.0, title="Two"),
            Chapter(start=20.0, end=30.0, title="Three"),
        ]
        output_path = temp_dir / "metadata.txt"

        write_ffmpeg_metadata(chapters, output_path)

        content = output_path.read_text()
        assert content.count("[CHAPTER]") == 3
        assert content.count("TIMEBASE=1/1000") == 3

    def test_write_ffmpeg_metadata_special_characters(self, temp_dir: Path) -> None:
        """Test that chapter titles with special characters are handled."""
        chapters = [
            Chapter(start=0.0, end=10.0, title="Test & Chapter: Part 1"),
        ]
        output_path = temp_dir / "metadata.txt"

        write_ffmpeg_metadata(chapters, output_path)

        content = output_path.read_text()
        assert "title=Test & Chapter: Part 1" in content


class TestEmbedChapters:
    """Tests for embed_chapters function."""

    def test_embed_chapters_success(self, mocker: MockerFixture, temp_dir: Path) -> None:
        """Test successful chapter embedding."""
        audio_path = temp_dir / "audio.mp3"
        audio_path.touch()
        chapters_file = temp_dir / "chapters.txt"
        chapters_file.touch()
        output_path = temp_dir / "output.mp3"

        mock_run = mocker.patch("chapterss.insert_chapters.subprocess.run")

        embed_chapters(audio_path, chapters_file, output_path)

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[0][0][0] == "ffmpeg"
        assert str(audio_path) in call_args[0][0]
        assert str(chapters_file) in call_args[0][0]
        assert str(output_path) in call_args[0][0]
        assert call_args[1]["check"] is True

    def test_embed_chapters_ffmpeg_not_found(self, mocker: MockerFixture, temp_dir: Path) -> None:
        """Test error when FFmpeg is not installed."""
        audio_path = temp_dir / "audio.mp3"
        audio_path.touch()
        chapters_file = temp_dir / "chapters.txt"
        chapters_file.touch()
        output_path = temp_dir / "output.mp3"

        mocker.patch("chapterss.insert_chapters.subprocess.run", side_effect=FileNotFoundError())

        with pytest.raises(RuntimeError, match="FFmpeg is not installed"):
            embed_chapters(audio_path, chapters_file, output_path)

    def test_embed_chapters_ffmpeg_error(self, mocker: MockerFixture, temp_dir: Path) -> None:
        """Test error when FFmpeg command fails."""
        audio_path = temp_dir / "audio.mp3"
        audio_path.touch()
        chapters_file = temp_dir / "chapters.txt"
        chapters_file.touch()
        output_path = temp_dir / "output.mp3"

        error = subprocess.CalledProcessError(1, "ffmpeg", stderr="Error message")
        mocker.patch("chapterss.insert_chapters.subprocess.run", side_effect=error)

        with pytest.raises(RuntimeError, match="Failed to embed chapters"):
            embed_chapters(audio_path, chapters_file, output_path)


class TestProcessEpisode:
    """Tests for process_episode function."""

    def test_process_episode_basic(self, mocker: MockerFixture, temp_dir: Path) -> None:
        """Test basic episode processing."""
        audio_path = temp_dir / "audio.mp3"
        audio_path.touch()
        markers_folder = temp_dir / "markers"
        markers_folder.mkdir()
        (markers_folder / "intro.wav").touch()
        output_path = temp_dir / "output.mp3"

        chapters = [
            Chapter(start=0.0, end=100.0, title="Intro"),
            Chapter(start=100.0, end=200.0, title="Chapter 1"),
        ]

        mocker.patch("chapterss.insert_chapters.detect_marked_chapters", return_value=chapters)
        mock_embed = mocker.patch("chapterss.insert_chapters.embed_chapters")
        mocker.patch("chapterss.insert_chapters.write_ffmpeg_metadata")

        result = process_episode(audio_path, markers_folder, output_path)

        assert result == output_path
        mock_embed.assert_called_once()

    def test_process_episode_no_chapters(self, mocker: MockerFixture, temp_dir: Path) -> None:
        """Test episode processing when no chapters are detected."""
        audio_path = temp_dir / "audio.mp3"
        audio_path.touch()
        markers_folder = temp_dir / "markers"
        markers_folder.mkdir()
        (markers_folder / "intro.wav").touch()
        output_path = temp_dir / "output.mp3"

        mocker.patch("chapterss.insert_chapters.detect_marked_chapters", return_value=[])
        mock_embed = mocker.patch("chapterss.insert_chapters.embed_chapters")

        result = process_episode(audio_path, markers_folder, output_path)

        assert result is None
        mock_embed.assert_not_called()

    def test_process_episode_invalid_markers_folder(self, temp_dir: Path) -> None:
        """Test error when markers folder doesn't exist."""
        audio_path = temp_dir / "audio.mp3"
        audio_path.touch()
        markers_folder = temp_dir / "nonexistent"
        output_path = temp_dir / "output.mp3"

        with pytest.raises(ValueError, match="Markers path must be a directory"):
            process_episode(audio_path, markers_folder, output_path)

    def test_process_episode_no_marker_files(self, temp_dir: Path) -> None:
        """Test error when markers folder is empty."""
        audio_path = temp_dir / "audio.mp3"
        audio_path.touch()
        markers_folder = temp_dir / "markers"
        markers_folder.mkdir()
        output_path = temp_dir / "output.mp3"

        with pytest.raises(ValueError, match="No audio files found"):
            process_episode(audio_path, markers_folder, output_path)

    def test_process_episode_filters_marker_extensions(self, mocker: MockerFixture, temp_dir: Path) -> None:
        """Test that only audio files are used as markers."""
        audio_path = temp_dir / "audio.mp3"
        audio_path.touch()
        markers_folder = temp_dir / "markers"
        markers_folder.mkdir()
        (markers_folder / "intro.wav").touch()
        (markers_folder / "outro.mp3").touch()
        (markers_folder / "readme.txt").touch()
        (markers_folder / "config.yaml").touch()
        output_path = temp_dir / "output.mp3"

        chapters = [Chapter(start=0.0, end=100.0, title="Test")]

        mock_detect = mocker.patch("chapterss.insert_chapters.detect_marked_chapters", return_value=chapters)
        mocker.patch("chapterss.insert_chapters.embed_chapters")
        mocker.patch("chapterss.insert_chapters.write_ffmpeg_metadata")

        process_episode(audio_path, markers_folder, output_path)

        # Check that only audio files were passed
        call_args = mock_detect.call_args
        marker_paths = call_args[0][1]
        assert len(marker_paths) == 2
        assert all(path.suffix.lower() in {".wav", ".mp3"} for path in marker_paths.values())

    def test_process_episode_with_transcribe(self, mocker: MockerFixture, temp_dir: Path) -> None:
        """Test episode processing with transcription enabled."""
        audio_path = temp_dir / "audio.mp3"
        audio_path.touch()
        markers_folder = temp_dir / "markers"
        markers_folder.mkdir()
        (markers_folder / "intro.wav").touch()
        output_path = temp_dir / "output.mp3"

        chapters = [
            Chapter(start=0.0, end=100.0, title="Original Title"),
        ]

        mocker.patch("chapterss.insert_chapters.detect_marked_chapters", return_value=chapters)
        mock_transcribe = mocker.patch("chapterss.insert_chapters.transcribe_func", return_value="Transcribed Title")
        mocker.patch("chapterss.insert_chapters.embed_chapters")
        mocker.patch("chapterss.insert_chapters.write_ffmpeg_metadata")

        process_episode(audio_path, markers_folder, output_path, transcribe=True)

        mock_transcribe.assert_called_once()
        # Verify chapter title was updated
        assert chapters[0].title == "Transcribed Title"

    def test_process_episode_with_summarise(self, mocker: MockerFixture, temp_dir: Path) -> None:
        """Test episode processing with summarization enabled."""
        audio_path = temp_dir / "audio.mp3"
        audio_path.touch()
        markers_folder = temp_dir / "markers"
        markers_folder.mkdir()
        (markers_folder / "intro.wav").touch()
        output_path = temp_dir / "output.mp3"

        chapters = [
            Chapter(start=0.0, end=100.0, title="Long Title That Needs Summarization"),
        ]

        mocker.patch("chapterss.insert_chapters.detect_marked_chapters", return_value=chapters)
        mock_summarise = mocker.patch("chapterss.insert_chapters.summarise_func", return_value="Short Title")
        mocker.patch("chapterss.insert_chapters.embed_chapters")
        mocker.patch("chapterss.insert_chapters.write_ffmpeg_metadata")

        process_episode(audio_path, markers_folder, output_path, summarise=True)

        mock_summarise.assert_called_once()
        assert chapters[0].title == "Short Title"

    def test_process_episode_transcribe_duration_limit(self, mocker: MockerFixture, temp_dir: Path) -> None:
        """Test that transcription duration is limited to 30 seconds."""
        audio_path = temp_dir / "audio.mp3"
        audio_path.touch()
        markers_folder = temp_dir / "markers"
        markers_folder.mkdir()
        (markers_folder / "intro.wav").touch()
        output_path = temp_dir / "output.mp3"

        # Chapter longer than 30 seconds
        chapters = [
            Chapter(start=0.0, end=60.0, title="Long Chapter"),
        ]

        mocker.patch("chapterss.insert_chapters.detect_marked_chapters", return_value=chapters)
        mock_transcribe = mocker.patch("chapterss.insert_chapters.transcribe_func", return_value="Transcribed")
        mocker.patch("chapterss.insert_chapters.embed_chapters")
        mocker.patch("chapterss.insert_chapters.write_ffmpeg_metadata")

        process_episode(audio_path, markers_folder, output_path, transcribe=True)

        # Verify transcription was called with max 30 seconds
        call_args = mock_transcribe.call_args
        duration = call_args[0][2]
        assert duration == 30.0

    def test_process_episode_cleans_up_metadata_file(self, mocker: MockerFixture, temp_dir: Path) -> None:
        """Test that metadata file is cleaned up after embedding."""
        audio_path = temp_dir / "audio.mp3"
        audio_path.touch()
        markers_folder = temp_dir / "markers"
        markers_folder.mkdir()
        (markers_folder / "intro.wav").touch()
        output_path = temp_dir / "output.mp3"

        chapters = [Chapter(start=0.0, end=100.0, title="Test")]

        mocker.patch("chapterss.insert_chapters.detect_marked_chapters", return_value=chapters)
        mocker.patch("chapterss.insert_chapters.embed_chapters")

        # Track write_ffmpeg_metadata calls to get the metadata file path
        metadata_path = None

        def track_write(chapters, path):
            nonlocal metadata_path
            metadata_path = path
            path.touch()

        mocker.patch("chapterss.insert_chapters.write_ffmpeg_metadata", side_effect=track_write)

        process_episode(audio_path, markers_folder, output_path)

        # Metadata file should be deleted
        if metadata_path:
            assert not metadata_path.exists()

    def test_process_episode_custom_threshold(self, mocker: MockerFixture, temp_dir: Path) -> None:
        """Test episode processing with custom detection threshold."""
        audio_path = temp_dir / "audio.mp3"
        audio_path.touch()
        markers_folder = temp_dir / "markers"
        markers_folder.mkdir()
        (markers_folder / "intro.wav").touch()
        output_path = temp_dir / "output.mp3"

        chapters = [Chapter(start=0.0, end=100.0, title="Test")]

        mock_detect = mocker.patch("chapterss.insert_chapters.detect_marked_chapters", return_value=chapters)
        mocker.patch("chapterss.insert_chapters.embed_chapters")
        mocker.patch("chapterss.insert_chapters.write_ffmpeg_metadata")

        process_episode(audio_path, markers_folder, output_path, threshold=0.9, min_gap=10.0)

        # Verify custom parameters were passed
        call_args = mock_detect.call_args
        assert call_args[1]["threshold"] == 0.9
        assert call_args[1]["min_gap"] == 10.0

    def test_process_episode_sanitizes_marker_names(self, mocker: MockerFixture, temp_dir: Path) -> None:
        """Test that marker names are sanitized."""
        audio_path = temp_dir / "audio.mp3"
        audio_path.touch()
        markers_folder = temp_dir / "markers"
        markers_folder.mkdir()
        (markers_folder / "intro@#$.wav").touch()
        (markers_folder / "chapter-1.mp3").touch()
        output_path = temp_dir / "output.mp3"

        chapters = [Chapter(start=0.0, end=100.0, title="Test")]

        mock_detect = mocker.patch("chapterss.insert_chapters.detect_marked_chapters", return_value=chapters)
        mocker.patch("chapterss.insert_chapters.embed_chapters")
        mocker.patch("chapterss.insert_chapters.write_ffmpeg_metadata")

        process_episode(audio_path, markers_folder, output_path)

        # Check that sanitized names were used
        call_args = mock_detect.call_args
        marker_paths = call_args[0][1]
        assert "intro" in marker_paths
        assert "chapter-1" in marker_paths
        # Special characters should be removed
        assert not any("@" in name or "#" in name or "$" in name for name in marker_paths.keys())
