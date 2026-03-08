import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
from numpy.typing import NDArray
from scipy.signal import correlate

log: logging.Logger = logging.getLogger(__name__)


@dataclass
class Marker:
    time: float
    name: str
    confidence: float
    offset: float = 0.0  # Length of the marker in seconds


@dataclass
class Chapter:
    start: float
    end: float
    title: str


def load_audio_features(path: Path, sr: int = 22050, hop_length: int = 512) -> tuple[NDArray[np.floating], float]:
    duration = librosa.get_duration(path=path)
    audio, _ = librosa.load(path, sr=sr)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20, hop_length=hop_length)
    mfcc_delta = librosa.feature.delta(mfcc)
    features = np.vstack([mfcc, mfcc_delta])
    features -= np.mean(features, axis=1, keepdims=True)
    features /= np.std(features, axis=1, keepdims=True) + 1e-6
    return features, duration


def detect_markers(
    audio_path: Path,
    marker_paths: Dict[str, Path],
    threshold: float = 0.75,
    min_gap: float = 8.0,
) -> List[Marker]:
    log.debug(f"Detecting {len(marker_paths)} markers in {audio_path}...")

    hop: int = 512
    sr: int = 22050

    log.debug("Loading audio features for main audio...")
    audio_features, audio_dur = load_audio_features(audio_path, sr=sr, hop_length=hop)

    all_detections: List[Marker] = []

    # Process each marker type
    for marker_name, marker_path in marker_paths.items():
        log.debug(f"Processing marker '{marker_name}' from {marker_path}...")
        marker_features, marker_dur = load_audio_features(marker_path, sr=sr, hop_length=hop)

        # Compute correlation for each feature dimension and average
        correlations: List[NDArray[np.floating]] = []
        for i in range(audio_features.shape[0]):
            corr = correlate(audio_features[i], marker_features[i], mode="valid")
            # Normalize each dimension
            if np.max(np.abs(corr)) > 0:
                corr = corr / np.max(np.abs(corr))
            correlations.append(corr)

        # Average across all MFCC dimensions for final correlation
        avg_corr = np.mean(correlations, axis=0)

        # Normalize to [0, 1]
        if np.max(avg_corr) > 0:
            avg_corr = (avg_corr - np.min(avg_corr)) / (np.max(avg_corr) - np.min(avg_corr))

        # Find peaks above threshold
        peaks = np.where(avg_corr > threshold)[0]
        times = peaks * hop / sr
        confidences = avg_corr[peaks]

        # Add all detections for this marker
        for t, conf in zip(times, confidences):
            all_detections.append(Marker(time=t, name=marker_name, confidence=conf, offset=marker_dur))

    # Sort by time
    log.debug(f"Deduplicating {len(all_detections)} markers...")
    all_detections.sort(key=lambda x: x.time)
    cleaned: List[Marker] = []
    for marker in all_detections:
        if not cleaned or abs(marker.time - cleaned[-1].time) > min_gap:
            cleaned.append(marker)
        elif marker.confidence > cleaned[-1].confidence:
            # Replace with higher confidence detection
            cleaned[-1] = marker

    log.debug(f"Detected {len(cleaned)} markers")
    return cleaned


def detect_marked_chapters(
    audio_path: Path,
    marker_paths: Dict[str, Path],
    threshold: float = 0.75,
    min_gap: float = 8.0,
    intro_threshold: float = 2.0,
) -> List[Chapter]:
    audio_duration = librosa.get_duration(path=audio_path)
    markers = detect_markers(audio_path, marker_paths, threshold, min_gap)

    if not markers:
        return []

    chapters: List[Chapter] = []

    # Add intro chapter if there's significant time before the first marker
    if markers[0].time > intro_threshold:
        chapters.append(Chapter(start=0.0, end=markers[0].time, title="Intro"))

    # Create chapters between markers
    for i, marker in enumerate(markers):
        chapter_start = marker.time + marker.offset

        # Chapter ends at the start of the next marker, or at the end of the audio file
        if i + 1 < len(markers):
            chapter_end = markers[i + 1].time
        else:
            chapter_end = audio_duration

        chapters.append(Chapter(start=chapter_start, end=chapter_end, title=marker.name))

    return chapters


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Detect markers in audio files")
    parser.add_argument("audio", type=Path, help="Path to the audio file")
    parser.add_argument("markers", type=Path, help="Folder containing marker audio files")
    parser.add_argument("--threshold", type=float, default=0.95, help="Correlation threshold")
    parser.add_argument("--min-gap", type=float, default=8.0, help="Minimum gap between markers in seconds")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    args: argparse.Namespace = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("chapterss").setLevel(log_level)

    marker_paths: Dict[str, Path] = {}

    for marker in sorted(args.markers.glob("*")):
        name: str = marker.stem
        safe_name: str = "".join(c for c in name if c.isalnum() or c in "_ -")
        if safe_name:
            marker_paths[safe_name] = marker

    if not marker_paths:
        parser.error(f"No audio files found in {args.markers}")

    log.debug(f"Loading audio: {args.audio}")
    log.debug(f"Markers to detect: {', '.join(marker_paths.keys())}")

    markers: List[Marker] = detect_markers(args.audio, marker_paths, args.threshold, args.min_gap)

    print(f"Found {len(markers)} markers:")
    for marker in markers:
        mins: int = int(marker.time // 60)
        secs: int = int(marker.time % 60)
        print(f"  {mins:02d}:{secs:02d} ({marker.time:.2f}s) - {marker.name} - {marker.confidence * 100:.0f}%")
