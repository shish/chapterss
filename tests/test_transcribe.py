from pathlib import Path
from unittest.mock import MagicMock

from pytest_mock import MockerFixture

from chapterss.transcribe import transcribe


class TestTranscribe:
    """Tests for transcribe function."""

    def test_transcribe_basic(self, mocker: MockerFixture, temp_audio_file: Path) -> None:
        """Test basic transcription."""
        # Mock the Model class
        mock_model_instance = MagicMock()
        mock_segment1 = MagicMock()
        mock_segment1.text = "Hello"
        mock_segment2 = MagicMock()
        mock_segment2.text = "World"
        mock_model_instance.transcribe.return_value = [mock_segment1, mock_segment2]

        mock_model_class = mocker.patch("chapterss.transcribe.Model", return_value=mock_model_instance)

        # Reset global model state
        import chapterss.transcribe

        chapterss.transcribe.model = None

        result = transcribe(temp_audio_file, start=0.0, duration=10.0)

        assert result == "Hello World"
        mock_model_class.assert_called_once()
        mock_model_instance.transcribe.assert_called_once_with(
            str(temp_audio_file),
            offset_ms=0,
            duration_ms=10000,
        )
