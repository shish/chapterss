"""Tests for extract_clips module."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import srt
from pytest_mock import MockerFixture

from chapterss.extract_clips import extract_clips


class TestExtractClips:
    """Tests for extract_clips function."""

    def test_extract_clips_basic(self, mocker: MockerFixture, temp_dir: Path) -> None:
        """Test basic clip extraction."""
        audio_path = temp_dir / "test.mp3"
        srt_path = temp_dir / "test.srt"
        output_dir = temp_dir / "output"

        # Create test SRT content
        srt_content = """1
00:00:00,000 --> 00:00:05,000
Intro Music

2
00:00:05,000 --> 00:00:10,000
Outro Music
"""
        srt_path.write_text(srt_content)

        # Mock AudioSegment
        mock_audio = MagicMock()
        mock_audio.__len__ = MagicMock(return_value=10000)  # 10 seconds
        mock_segment = MagicMock()
        mock_audio.__getitem__ = MagicMock(return_value=mock_segment)

        mock_from_file = mocker.patch("chapterss.extract_clips.AudioSegment.from_file", return_value=mock_audio)

        extract_clips(audio_path, srt_path, output_dir)

        # Verify AudioSegment.from_file was called
        mock_from_file.assert_called_once_with(str(audio_path))

        # Verify output directory was created
        assert output_dir.exists()

        # Verify export was called for each segment
        assert mock_segment.export.call_count == 2

    def test_extract_clips_sanitizes_filenames(self, mocker: MockerFixture, temp_dir: Path) -> None:
        """Test that filenames are sanitized."""
        audio_path = temp_dir / "test.mp3"
        srt_path = temp_dir / "test.srt"
        output_dir = temp_dir / "output"

        # Create SRT with special characters
        srt_content = """1
00:00:00,000 --> 00:00:05,000
Hello/World: Test!
"""
        srt_path.write_text(srt_content)

        mock_audio = MagicMock()
        mock_audio.__len__ = MagicMock(return_value=10000)
        mock_segment = MagicMock()
        mock_audio.__getitem__ = MagicMock(return_value=mock_segment)

        mocker.patch("chapterss.extract_clips.AudioSegment.from_file", return_value=mock_audio)

        extract_clips(audio_path, srt_path, output_dir)

        # Check that export was called with sanitized filename
        call_args = mock_segment.export.call_args[0][0]
        assert isinstance(call_args, Path)
        # Filename should have special chars replaced with underscores
        assert "Hello_World__Test_" in str(call_args)

    def test_extract_clips_correct_timing(self, mocker: MockerFixture, temp_dir: Path) -> None:
        """Test that clips are extracted with correct timing."""
        audio_path = temp_dir / "test.mp3"
        srt_path = temp_dir / "test.srt"
        output_dir = temp_dir / "output"

        srt_content = """1
00:00:10,500 --> 00:00:15,750
Test Segment
"""
        srt_path.write_text(srt_content)

        mock_audio = MagicMock()
        mock_audio.__len__ = MagicMock(return_value=20000)

        # Track what slice was requested
        slices = []

        def track_slice(self, key):
            slices.append(key)
            return MagicMock()

        mock_audio.__getitem__ = track_slice

        mocker.patch("chapterss.extract_clips.AudioSegment.from_file", return_value=mock_audio)

        extract_clips(audio_path, srt_path, output_dir)

        # Verify the correct time slice was used
        assert len(slices) == 1
        # Start: 10.5s * 1000 = 10500ms, End: 15.75s * 1000 = 15750ms
        assert slices[0] == slice(10500, 15750)

    def test_extract_clips_empty_srt(self, mocker: MockerFixture, temp_dir: Path) -> None:
        """Test extraction with empty SRT file."""
        audio_path = temp_dir / "test.mp3"
        srt_path = temp_dir / "test.srt"
        output_dir = temp_dir / "output"

        srt_path.write_text("")

        mock_audio = MagicMock()
        mock_audio.__len__ = MagicMock(return_value=10000)

        mock_from_file = mocker.patch("chapterss.extract_clips.AudioSegment.from_file", return_value=mock_audio)

        extract_clips(audio_path, srt_path, output_dir)

        # Audio should still be loaded
        mock_from_file.assert_called_once()

        # But no exports should happen
        assert not hasattr(mock_audio, "export") or mock_audio.export.call_count == 0

    def test_extract_clips_multiple_segments(self, mocker: MockerFixture, temp_dir: Path) -> None:
        """Test extraction with multiple segments."""
        audio_path = temp_dir / "test.mp3"
        srt_path = temp_dir / "test.srt"
        output_dir = temp_dir / "output"

        srt_content = """1
00:00:00,000 --> 00:00:02,000
Segment One

2
00:00:02,000 --> 00:00:04,000
Segment Two

3
00:00:04,000 --> 00:00:06,000
Segment Three
"""
        srt_path.write_text(srt_content)

        mock_audio = MagicMock()
        mock_audio.__len__ = MagicMock(return_value=10000)
        mock_segment = MagicMock()
        mock_audio.__getitem__ = MagicMock(return_value=mock_segment)

        mocker.patch("chapterss.extract_clips.AudioSegment.from_file", return_value=mock_audio)

        extract_clips(audio_path, srt_path, output_dir)

        # Should extract 3 segments
        assert mock_segment.export.call_count == 3

    def test_extract_clips_creates_output_directory(self, mocker: MockerFixture, temp_dir: Path) -> None:
        """Test that output directory is created if it doesn't exist."""
        audio_path = temp_dir / "test.mp3"
        srt_path = temp_dir / "test.srt"
        output_dir = temp_dir / "nested" / "output" / "dir"

        assert not output_dir.exists()

        srt_content = """1
00:00:00,000 --> 00:00:05,000
Test
"""
        srt_path.write_text(srt_content)

        mock_audio = MagicMock()
        mock_audio.__len__ = MagicMock(return_value=10000)
        mock_segment = MagicMock()
        mock_audio.__getitem__ = MagicMock(return_value=mock_segment)

        mocker.patch("chapterss.extract_clips.AudioSegment.from_file", return_value=mock_audio)

        extract_clips(audio_path, srt_path, output_dir)

        # Output directory should be created
        assert output_dir.exists()
