from unittest.mock import MagicMock

from anthropic.types import Message, TextBlock, Usage
from pytest_mock import MockerFixture

from chapterss.summarise import summarise


class TestSummarise:
    """Tests for summarise function."""

    def test_summarise_basic(self, mocker: MockerFixture) -> None:
        """Test basic text summarization."""
        # Mock Anthropic client
        mock_client_instance = MagicMock()
        mock_response = Message(
            id="msg_123",
            type="message",
            role="assistant",
            content=[TextBlock(type="text", text="Brief summary")],
            model="claude-3-haiku-20240307",
            stop_reason="end_turn",
            stop_sequence=None,
            usage=Usage(input_tokens=10, output_tokens=5),
        )
        mock_client_instance.messages.create.return_value = mock_response

        mock_anthropic = mocker.patch("chapterss.summarise.Anthropic", return_value=mock_client_instance)

        result = summarise("This is a long text that needs to be summarized.", max_words=5, api_key="test-key")

        assert result == "Brief summary"
        mock_anthropic.assert_called_once_with(api_key="test-key")
        mock_client_instance.messages.create.assert_called_once()
