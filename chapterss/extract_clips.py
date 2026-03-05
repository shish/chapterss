import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

from pydub import AudioSegment

log: logging.Logger = logging.getLogger(__name__)


@dataclass
class SRTSegment:
    name: str
    filename: str
    start_ms: int
    end_ms: int


def parse_srt_timestamp(timestamp: str) -> int:
    """Parse SRT timestamp (HH:MM:SS,mmm) to milliseconds."""
    try:
        time_part, ms_part = timestamp.split(",")
        h, m, s = map(int, time_part.split(":"))
        ms: int = int(ms_part)

        # Validate ranges
        if not (0 <= h <= 99 and 0 <= m <= 59 and 0 <= s <= 59 and 0 <= ms <= 999):
            raise ValueError(f"Timestamp values out of range: {timestamp}")

        return (h * 3600 + m * 60 + s) * 1000 + ms
    except (ValueError, AttributeError) as e:
        raise ValueError(f"Invalid timestamp format: {timestamp}") from e


def parse_srt(srt_path: Path) -> List[SRTSegment]:
    """Parse SRT file and extract segments with names."""
    segments: List[SRTSegment] = []

    # Validate file size to prevent memory issues
    max_srt_size: int = 10 * 1024 * 1024  # 10MB
    if srt_path.stat().st_size > max_srt_size:
        raise ValueError(f"SRT file too large (max {max_srt_size} bytes)")

    content: str
    try:
        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        # Try with latin-1 as fallback
        with open(srt_path, "r", encoding="latin-1") as f:
            content = f.read()

    # Split by double newlines to get each subtitle block
    blocks: List[str] = content.strip().split("\n\n")

    # Limit number of blocks to prevent DoS
    max_blocks: int = 10000
    if len(blocks) > max_blocks:
        raise ValueError(f"Too many SRT blocks (max {max_blocks})")

    for block in blocks:
        lines: List[str] = block.strip().split("\n")
        if len(lines) < 3:
            continue

        # Line 0: index number
        # Line 1: timestamp range
        # Line 2+: text content
        timestamp_line: str = lines[1]
        text: str = " ".join(lines[2:]).strip()

        # Limit text length
        if len(text) > 1000:
            log.warning(f"Truncating long text: {text[:50]}...")
            text = text[:1000]

        # Parse timestamps
        match: Union[re.Match[str], None] = re.match(
            r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})", timestamp_line
        )
        if match:
            start_ts, end_ts = match.groups()
            try:
                start_ms = parse_srt_timestamp(start_ts)
                end_ms = parse_srt_timestamp(end_ts)
            except ValueError as e:
                log.warning(f"Skipping invalid timestamp: {e}")
                continue

            # Validate timing makes sense
            if end_ms <= start_ms:
                log.warning(f"Skipping segment with invalid timing: {start_ts} to {end_ts}")
                continue

            if end_ms - start_ms > 3600000:  # 1 hour
                log.warning("Skipping segment longer than 1 hour")
                continue

            # Convert text to filename-safe format
            filename: str = text.replace(" ", "_").replace("/", "_")
            # Remove any characters that aren't alphanumeric, underscore, or hyphen
            filename = re.sub(r"[^\w\-]", "", filename)

            # Ensure filename isn't empty and isn't too long
            if not filename:
                filename = f"segment_{len(segments)}"
            filename = filename[:100]  # Limit length

            segments.append(
                SRTSegment(
                    name=text,
                    filename=filename,
                    start_ms=start_ms,
                    end_ms=end_ms,
                )
            )

    return segments


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Extract named audio segments from SRT file")
    parser.add_argument("audio", type=Path, help="Path to the audio file (e.g., lateral.mp3)")
    parser.add_argument("srt", type=Path, help="Path to the SRT file with labeled segments")
    parser.add_argument("-o", "--output-dir", type=Path, default=".", help="Output directory for extracted segments")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    args: argparse.Namespace = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s %(name)s %(message)s")

    try:
        segments: List[SRTSegment] = parse_srt(args.srt)
        log.debug(f"Found {len(segments)} segments in {args.srt}")
        if not segments:
            log.error("No valid segments found in SRT file")
            exit(1)

        # Load audio file once
        log.debug(f"Loading audio file: {args.audio}")
        audio: AudioSegment = AudioSegment.from_file(str(args.audio))
        log.debug(f"Audio loaded: {len(audio)}ms duration")

        # Create output directory if it doesn't exist
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Extract each segment
        for segment in segments:
            output_path: Path = args.output_dir / f"{segment.filename}.wav"
            start_sec: float = segment.start_ms / 1000
            end_sec: float = segment.end_ms / 1000
            duration_sec: float = (segment.end_ms - segment.start_ms) / 1000
            log.debug(f"Extracting '{segment.name}': {start_sec:.2f}s - {end_sec:.2f}s ({duration_sec:.2f}s)")
            audio[segment.start_ms : segment.end_ms].export(output_path, format="wav")
            log.debug(f"  -> Saved to {output_path}")

        print(f"Extracted {len(segments)} clips to {args.output_dir}")

    except ValueError as e:
        log.error(f"Validation error: {e}")
        exit(1)
    except Exception as e:
        log.error(f"Error: {e}")
        exit(1)
