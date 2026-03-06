import argparse
import logging
import tempfile
from pathlib import Path

import whisper
from pydub import AudioSegment

log: logging.Logger = logging.getLogger(__name__)


def transcribe(audio_path: Path, start: float, duration: float) -> str:
    log.info(f"Transcribing {audio_path} from {start:.2f}s to {start + duration:.2f}s")

    log.debug(f"Loading audio file: {audio_path}")
    audio = AudioSegment.from_file(audio_path)
    start_ms = int(start * 1000)
    end_ms = int((start + duration) * 1000)
    segment = audio[start_ms:end_ms]

    log.debug("Exporting segment")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = temp_file.name
        segment.export(temp_path, format="wav")

    try:
        log.debug("Load model")
        model = whisper.load_model("base")
        log.debug("Transcribing segment")
        result = model.transcribe(temp_path)
        log.debug("Transcription complete")
        return result["text"].strip()
    finally:
        Path(temp_path).unlink(missing_ok=True)


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Transcribe audio segments")
    parser.add_argument("audio", type=Path, help="Path to the audio file (e.g., lateral.mp3)")
    parser.add_argument("start", type=float, help="Start time of the segment in seconds")
    parser.add_argument("duration", type=float, help="Duration of the segment in seconds")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    args: argparse.Namespace = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("chapterss").setLevel(log_level)

    transcription: str = transcribe(args.audio, args.start, args.duration)
    print(transcription)
