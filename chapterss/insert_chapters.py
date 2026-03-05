import argparse
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from .detect_markers import Chapter, detect_markers

log: logging.Logger = logging.getLogger(__name__)


def write_ffmpeg_metadata(chapters: List[Chapter], output_path: Path) -> None:
    """Write chapter metadata in FFmpeg format."""
    with open(output_path, "w") as f:
        f.write(";FFMETADATA1\n")
        for i, chapter in enumerate(chapters):
            # Convert time to milliseconds for FFmpeg
            start_ms: int = int(chapter.time * 1000)
            end_ms: int

            # Use next chapter start or add 1 second as end time
            if i + 1 < len(chapters):
                end_ms = int(chapters[i + 1].time * 1000)
            else:
                end_ms = start_ms + 1000

            f.write("\n[CHAPTER]\n")
            f.write("TIMEBASE=1/1000\n")
            f.write(f"START={start_ms}\n")
            f.write(f"END={end_ms}\n")
            f.write(f"title={chapter.name}\n")


def embed_chapters(audio_path: Path, chapters_file: Path, output_path: Path) -> None:
    """Embed chapter metadata into audio file using FFmpeg."""
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(audio_path),
                "-i",
                str(chapters_file),
                "-map_metadata",
                "1",
                "-codec",
                "copy",
                "-y",  # Overwrite output file if it exists
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        log.error(f"FFmpeg failed: {e.stderr}")
        raise RuntimeError(f"Failed to embed chapters: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("FFmpeg is not installed or not in PATH")


def process_episode(
    audio_path: Path,
    markers_folder: Path,
    output_path: Path,
    threshold: float = 0.85,
    min_gap: float = 8.0,
) -> Optional[Path]:
    """
    Process a podcast episode: detect markers and embed chapters.

    Args:
        audio_path: Path to the audio file
        markers_folder: Path to folder containing marker audio files
        output_path: Path for the output audio file with chapters
        threshold: Detection threshold (0.0-1.0)
        min_gap: Minimum gap between detections in seconds
    """
    # Validate threshold and min_gap
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
    if min_gap < 0:
        raise ValueError(f"min_gap must be non-negative, got {min_gap}")

    log.debug(f"Processing: {audio_path}")
    log.debug(f"Markers folder: {markers_folder}")
    log.debug(f"Output: {output_path}")

    # Load marker files
    if not markers_folder.is_dir():
        raise ValueError(f"Markers path must be a directory: {markers_folder}")

    marker_paths: Dict[str, Path] = {}
    allowed_extensions: set[str] = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

    for audio_file in sorted(markers_folder.glob("*")):
        # Ensure we only process files within the markers folder (no symlinks to outside)
        try:
            audio_file_resolved: Path = audio_file.resolve()
            audio_file_resolved.relative_to(markers_folder.resolve())
        except ValueError, RuntimeError:
            log.warning(f"Skipping file outside markers folder: {audio_file}")
            continue

        if audio_file.suffix.lower() in allowed_extensions:
            name: str = audio_file.stem
            # Sanitize marker name to prevent injection in chapter titles
            safe_name: str = "".join(c for c in name if c.isalnum() or c in "_ -")
            if safe_name:
                marker_paths[safe_name] = audio_file

    if not marker_paths:
        raise ValueError(f"No audio files found in {markers_folder}")

    log.debug(f"Loaded {len(marker_paths)} markers: {', '.join(marker_paths.keys())}")

    # Detect markers
    log.debug("Detecting markers...")

    chapters: List[Chapter] = detect_markers(audio_path, marker_paths, threshold=threshold, min_gap=min_gap)

    log.info(f"Found {len(chapters)} chapters:")
    for chapter in chapters:
        mins: int = int(chapter.time // 60)
        secs: int = int(chapter.time % 60)
        log.info(f"  {mins:02d}:{secs:02d} - {chapter.name} ({chapter.confidence * 100:.0f}%)")

    if not chapters:
        log.info("Warning: No chapters detected!")
        return None

    # Embed chapters using FFmpeg
    metadata_file: Path = output_path.with_suffix(".metadata.txt")
    log.debug(f"Writing metadata to: {metadata_file}")
    write_ffmpeg_metadata(chapters, metadata_file)
    log.debug(f"Embedding chapters into: {output_path}")
    embed_chapters(audio_path, metadata_file, output_path)
    log.info(f"✓ Success! Created: {output_path}")
    try:
        metadata_file.unlink()
    except OSError as e:
        log.warning(f"Failed to clean up metadata file: {e}")

    return output_path


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Process a podcast episode and embed chapter markers"
    )
    parser.add_argument("audio", type=Path, help="Path to the audio file")
    parser.add_argument("markers", type=Path, help="Folder containing marker audio files")
    parser.add_argument("output", type=Path, help="Path for the output audio file with chapters")
    parser.add_argument("--threshold", type=float, default=0.85, help="Detection threshold")
    parser.add_argument("--min-gap", type=float, default=8.0, help="Minimum gap between chapters in seconds")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    args: argparse.Namespace = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s %(name)s %(message)s")

    try:
        process_episode(
            args.audio,
            args.markers,
            args.output,
            threshold=args.threshold,
            min_gap=args.min_gap,
        )
    except Exception as e:
        log.error(f"Error: {e}")
        exit(1)
