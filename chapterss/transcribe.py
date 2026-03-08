import argparse
import logging
from pathlib import Path

from pywhispercpp.model import Model

log: logging.Logger = logging.getLogger(__name__)

model = None


def transcribe(audio_path: Path, start: float, duration: float) -> str:
    global model

    if model is None:
        model_dir = Path("data") / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        log.debug(f"Load STT model from {model_dir}")
        model = Model("base", models_dir=str(model_dir))

    log.info(f"Transcribing {audio_path} from {start:.2f}s to {start + duration:.2f}s")
    result = model.transcribe(
        str(audio_path),
        offset_ms=int(start * 1000),
        duration_ms=int(duration * 1000),
    )
    log.debug("Transcription complete")
    return " ".join(seg.text for seg in result)


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
