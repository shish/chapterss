import argparse
import logging
import re
from pathlib import Path

import srt
from pydub import AudioSegment

log: logging.Logger = logging.getLogger(__name__)


def extract_clips(audio_path: Path, srt_path: Path, output_dir: Path) -> None:
    segments = list(srt.parse(srt_path.read_text()))
    log.debug(f"Found {len(segments)} segments in {srt_path}")

    log.debug(f"Loading audio file: {audio_path}")
    audio: AudioSegment = AudioSegment.from_file(str(audio_path))
    log.debug(f"Audio loaded: {len(audio)}ms duration")

    output_dir.mkdir(parents=True, exist_ok=True)
    for subtitle in segments:
        filename = re.sub(r"[^a-zA-Z0-9]", "_", subtitle.content.strip())
        start = subtitle.start.total_seconds()
        end = subtitle.end.total_seconds()
        duration = end - start

        output_path: Path = output_dir / f"{filename}.wav"
        log.debug(f"Extracting '{filename}': {start:.2f}s - {end:.2f}s ({duration:.2f}s)")
        audio[int(start * 1000) : int(end * 1000)].export(output_path, format="wav")
        log.debug(f"  -> Saved to {output_path}")

    print(f"Extracted {len(segments)} clips to {output_dir}")


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Extract named audio segments from SRT file")
    parser.add_argument("audio", type=Path, help="Path to the audio file (e.g., lateral.mp3)")
    parser.add_argument("srt", type=Path, help="Path to the SRT file with labeled segments")
    parser.add_argument("-o", "--output-dir", type=Path, default=".", help="Output directory for extracted segments")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    args: argparse.Namespace = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("chapterss").setLevel(log_level)

    extract_clips(args.audio, args.srt, args.output_dir)
