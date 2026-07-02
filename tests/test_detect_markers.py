"""Tests for detect_markers module."""

from pathlib import Path

import numpy as np
import pytest
from pytest_mock import MockerFixture

from chapterss.detect_markers import (
    Chapter,
    Marker,
    detect_marked_chapters,
    detect_markers,
    load_audio_features,
)


class TestLoadAudioFeatures:
    """Tests for load_audio_features function."""

    def test_load_audio_features(self, mocker: MockerFixture, temp_audio_file: Path) -> None:
        """Test loading audio features."""
        # Mock librosa functions
        mock_duration = mocker.patch("chapterss.detect_markers.librosa.get_duration", return_value=10.0)
        mock_load = mocker.patch(
            "chapterss.detect_markers.librosa.load",
            return_value=(np.random.rand(220500), 22050),
        )
        mock_mfcc = mocker.patch(
            "chapterss.detect_markers.librosa.feature.mfcc",
            return_value=np.random.rand(20, 100),
        )
        mock_delta = mocker.patch(
            "chapterss.detect_markers.librosa.feature.delta",
            return_value=np.random.rand(20, 100),
        )

        features, duration = load_audio_features(temp_audio_file)

        assert duration == 10.0
        assert features.shape == (40, 100)
        mock_duration.assert_called_once_with(path=temp_audio_file)
        mock_load.assert_called_once_with(temp_audio_file, sr=22050)
        mock_mfcc.assert_called_once()
        mock_delta.assert_called_once()


class TestDetectMarkers:
    """Tests for detect_markers function."""

    def test_detect_markers_basic(self, mocker: MockerFixture, temp_audio_file: Path, temp_marker_dir: Path) -> None:
        """Test basic marker detection."""
        # Mock load_audio_features to return predictable data
        audio_features = np.random.rand(40, 1000)
        marker_features = np.random.rand(40, 100)

        def mock_load_features(path: Path, sr: int = 22050, hop_length: int = 512) -> tuple:
            if path == temp_audio_file:
                return audio_features, 100.0
            else:
                return marker_features, 2.0

        mocker.patch("chapterss.detect_markers.load_audio_features", side_effect=mock_load_features)

        # Mock correlate to return some peaks
        def mock_correlate(a, b, mode):
            result = np.zeros(901)
            result[100] = 0.9
            result[500] = 0.85
            return result

        mocker.patch("chapterss.detect_markers.correlate", side_effect=mock_correlate)

        marker_paths = {
            "intro": temp_marker_dir / "intro.wav",
            "outro": temp_marker_dir / "outro.wav",
        }

        markers = detect_markers(temp_audio_file, marker_paths, threshold=0.75, min_gap=8.0)

        assert isinstance(markers, list)
        assert len(markers) >= 0
        for marker in markers:
            assert isinstance(marker, Marker)
            assert marker.name in ["intro", "outro"]

    def test_detect_markers_empty_result(
        self, mocker: MockerFixture, temp_audio_file: Path, temp_marker_dir: Path
    ) -> None:
        """Test marker detection with no matches."""
        audio_features = np.random.rand(40, 1000)
        marker_features = np.random.rand(40, 100)

        def mock_load_features(path: Path, sr: int = 22050, hop_length: int = 512) -> tuple:
            if path == temp_audio_file:
                return audio_features, 100.0
            else:
                return marker_features, 2.0

        mocker.patch("chapterss.detect_markers.load_audio_features", side_effect=mock_load_features)

        # Mock correlate to return low values
        mocker.patch("chapterss.detect_markers.correlate", return_value=np.zeros(901))

        marker_paths = {"intro": temp_marker_dir / "intro.wav"}

        markers = detect_markers(temp_audio_file, marker_paths, threshold=0.95, min_gap=8.0)

        assert markers == []

    def test_detect_markers_deduplication(
        self, mocker: MockerFixture, temp_audio_file: Path, temp_marker_dir: Path
    ) -> None:
        """Test that markers within min_gap are deduplicated."""
        audio_features = np.random.rand(40, 1000)
        marker_features = np.random.rand(40, 100)

        def mock_load_features(path: Path, sr: int = 22050, hop_length: int = 512) -> tuple:
            if path == temp_audio_file:
                return audio_features, 100.0
            else:
                return marker_features, 2.0

        mocker.patch("chapterss.detect_markers.load_audio_features", side_effect=mock_load_features)

        # Mock correlate to return peaks close together
        def mock_correlate(a, b, mode):
            result = np.zeros(901)
            result[100] = 0.9
            result[120] = 0.85  # Close to first peak
            return result

        mocker.patch("chapterss.detect_markers.correlate", side_effect=mock_correlate)

        marker_paths = {"intro": temp_marker_dir / "intro.wav"}

        markers = detect_markers(temp_audio_file, marker_paths, threshold=0.75, min_gap=8.0)

        # Should keep only the higher confidence one
        assert len(markers) <= 2


class TestDetectMarkedChapters:
    """Tests for detect_marked_chapters function."""

    def test_detect_marked_chapters_with_intro(
        self, mocker: MockerFixture, temp_audio_file: Path, temp_marker_dir: Path
    ) -> None:
        """Test chapter detection with intro."""
        # Mock librosa.get_duration
        mocker.patch("chapterss.detect_markers.librosa.get_duration", return_value=100.0)

        # Mock detect_markers to return predictable results
        markers = [
            Marker(time=10.0, name="Chapter 1", confidence=0.9, offset=1.0),
            Marker(time=50.0, name="Chapter 2", confidence=0.85, offset=1.0),
        ]
        mocker.patch("chapterss.detect_markers.detect_markers", return_value=markers)

        marker_paths = {"intro": temp_marker_dir / "intro.wav"}

        chapters = detect_marked_chapters(
            temp_audio_file, marker_paths, threshold=0.75, min_gap=8.0, intro_threshold=2.0
        )

        assert len(chapters) == 3
        assert chapters[0].title == "Intro"
        assert chapters[0].start == 0.0
        assert chapters[0].end == 10.0

        assert chapters[1].title == "Chapter 1"
        assert chapters[1].start == 11.0
        assert chapters[1].end == 50.0

        assert chapters[2].title == "Chapter 2"
        assert chapters[2].start == 51.0
        assert chapters[2].end == 100.0

    def test_detect_marked_chapters_no_intro(
        self, mocker: MockerFixture, temp_audio_file: Path, temp_marker_dir: Path
    ) -> None:
        """Test chapter detection without intro."""
        mocker.patch("chapterss.detect_markers.librosa.get_duration", return_value=100.0)

        markers = [
            Marker(time=1.0, name="Chapter 1", confidence=0.9, offset=0.5),
            Marker(time=50.0, name="Chapter 2", confidence=0.85, offset=0.5),
        ]
        mocker.patch("chapterss.detect_markers.detect_markers", return_value=markers)

        marker_paths = {"intro": temp_marker_dir / "intro.wav"}

        chapters = detect_marked_chapters(
            temp_audio_file, marker_paths, threshold=0.75, min_gap=8.0, intro_threshold=2.0
        )

        # No intro chapter because first marker is before intro_threshold
        assert len(chapters) == 2
        assert chapters[0].title == "Chapter 1"

    def test_detect_marked_chapters_no_markers(
        self, mocker: MockerFixture, temp_audio_file: Path, temp_marker_dir: Path
    ) -> None:
        """Test chapter detection with no markers found."""
        mocker.patch("chapterss.detect_markers.librosa.get_duration", return_value=100.0)
        mocker.patch("chapterss.detect_markers.detect_markers", return_value=[])

        marker_paths = {"intro": temp_marker_dir / "intro.wav"}

        chapters = detect_marked_chapters(temp_audio_file, marker_paths)

        assert chapters == []
