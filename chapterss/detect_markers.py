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
class Chapter:
    time: float
    name: str
    confidence: float


def load_audio_features(
    path: Path, sr: int = 22050, hop_length: int = 512
) -> tuple[NDArray[np.floating], NDArray[np.floating], int]:
    """Load audio and extract MFCC features for better matching."""

    # Limit audio file size to prevent memory issues (max 2GB)
    max_size: int = 2 * 1024 * 1024 * 1024
    if path.stat().st_size > max_size:
        raise ValueError(f"Audio file too large (max {max_size} bytes)")
    audio, _ = librosa.load(str(path), sr=sr)

    # Validate audio length (max 24 hours at 22050 Hz)
    max_samples: int = 24 * 3600 * sr
    if len(audio) > max_samples:
        raise ValueError("Audio too long (max 24 hours)")

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20, hop_length=hop_length)
    mfcc_delta = librosa.feature.delta(mfcc)
    features = np.vstack([mfcc, mfcc_delta])
    features -= np.mean(features, axis=1, keepdims=True)
    features /= np.std(features, axis=1, keepdims=True) + 1e-6
    return features, audio, sr


def detect_markers(
    audio_path: Path,
    marker_paths: Dict[str, Path],
    threshold: float = 0.75,
    min_gap: float = 8.0,
) -> List[Chapter]:
    """
    Detect multiple types of markers in an audio file using MFCC features.

    Args:
        audio_path: Path to the audio file to analyze
        marker_paths: Dict mapping marker names to their audio file paths
        threshold: Correlation threshold for detection (0.0-1.0)
        min_gap: Minimum gap between detections in seconds

    Returns:
        List of Chapter objects sorted by time
    """
    # Validate parameters
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
    if min_gap < 0:
        raise ValueError(f"min_gap must be non-negative, got {min_gap}")
    if not marker_paths:
        raise ValueError("No marker paths provided")
    if len(marker_paths) > 100:
        raise ValueError(f"Too many markers (max 100), got {len(marker_paths)}")

    hop: int = 512
    sr: int = 22050

    audio_features, audio, audio_sr = load_audio_features(audio_path, sr=sr, hop_length=hop)

    all_detections: List[Chapter] = []

    # Process each marker type
    for marker_name, marker_path in marker_paths.items():
        marker_features, marker_audio, marker_sr = load_audio_features(marker_path, sr=sr, hop_length=hop)

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
            all_detections.append(Chapter(time=t, name=marker_name, confidence=conf))

    # Sort by time
    all_detections.sort(key=lambda x: x.time)

    # Remove nearby duplicates, keeping the one with highest confidence
    cleaned: List[Chapter] = []
    for chapter in all_detections:
        if not cleaned or abs(chapter.time - cleaned[-1].time) > min_gap:
            cleaned.append(chapter)
        elif chapter.confidence > cleaned[-1].confidence:
            # Replace with higher confidence detection
            cleaned[-1] = chapter

    return cleaned


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Detect markers in audio files")
    parser.add_argument("audio", type=Path, help="Path to the audio file")
    parser.add_argument("markers", type=Path, help="Folder containing marker audio files")
    parser.add_argument("--threshold", type=float, default=0.85, help="Correlation threshold")
    parser.add_argument("--min-gap", type=float, default=8.0, help="Minimum gap between markers in seconds")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    args: argparse.Namespace = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s %(name)s %(message)s")

    try:
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

        markers: List[Chapter] = detect_markers(args.audio, marker_paths, args.threshold, args.min_gap)

        print(f"Found {len(markers)} markers:")
        for chapter in markers:
            mins: int = int(chapter.time // 60)
            secs: int = int(chapter.time % 60)
            print(f"  {mins:02d}:{secs:02d} ({chapter.time:.2f}s) - {chapter.name} - {chapter.confidence * 100:.0f}%")
    except ValueError as e:
        log.error(f"Validation error: {e}")
        exit(1)
    except Exception as e:
        log.exception(f"Error: {e}")
        exit(1)
