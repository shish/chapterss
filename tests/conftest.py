import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_audio_file(temp_dir: Path) -> Path:
    """Create a temporary audio file path."""
    audio_file = temp_dir / "test_audio.mp3"
    audio_file.touch()
    return audio_file


@pytest.fixture
def temp_marker_dir(temp_dir: Path) -> Path:
    """Create a temporary marker directory with some marker files."""
    marker_dir = temp_dir / "markers"
    marker_dir.mkdir()
    (marker_dir / "intro.wav").touch()
    (marker_dir / "outro.wav").touch()
    return marker_dir
