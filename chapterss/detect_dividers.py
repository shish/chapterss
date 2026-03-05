import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
from numpy.typing import NDArray
from scipy.signal import correlate

log: logging.Logger = logging.getLogger(__name__)


def load_audio_features(
    path: Path, sr: int = 22050, hop_length: int = 512
) -> Tuple[NDArray[np.floating], NDArray[np.floating], int]:
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


def detect_dividers(
    audio_path: Path,
    divider_paths: Dict[str, Path],
    threshold: float = 0.75,
    min_gap: float = 8.0,
) -> List[Tuple[float, str, float]]:
    """
    Detect multiple types of dividers in an audio file using MFCC features.

    Args:
        audio_path: Path to the audio file to analyze
        divider_paths: Dict mapping divider names to their audio file paths
        threshold: Correlation threshold for detection (0.0-1.0)
        min_gap: Minimum gap between detections in seconds

    Returns:
        List of tuples (time, divider_name, confidence) sorted by time
    """
    # Validate parameters
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
    if min_gap < 0:
        raise ValueError(f"min_gap must be non-negative, got {min_gap}")
    if not divider_paths:
        raise ValueError("No divider paths provided")
    if len(divider_paths) > 100:
        raise ValueError(f"Too many dividers (max 100), got {len(divider_paths)}")

    hop: int = 512
    sr: int = 22050

    audio_features, audio, audio_sr = load_audio_features(audio_path, sr=sr, hop_length=hop)

    all_detections: List[Tuple[float, str, float]] = []

    # Process each divider type
    for divider_name, divider_path in divider_paths.items():
        divider_features, divider_audio, divider_sr = load_audio_features(divider_path, sr=sr, hop_length=hop)

        # Compute correlation for each feature dimension and average
        correlations: List[NDArray[np.floating]] = []
        for i in range(audio_features.shape[0]):
            corr = correlate(audio_features[i], divider_features[i], mode="valid")
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

        # Add all detections for this divider
        for t, conf in zip(times, confidences):
            all_detections.append((t, divider_name, conf))

    # Sort by time
    all_detections.sort(key=lambda x: x[0])

    # Remove nearby duplicates, keeping the one with highest confidence
    cleaned: List[Tuple[float, str, float]] = []
    for time, name, conf in all_detections:
        if not cleaned or abs(time - cleaned[-1][0]) > min_gap:
            cleaned.append((time, name, conf))
        elif conf > cleaned[-1][2]:
            # Replace with higher confidence detection
            cleaned[-1] = (time, name, conf)

    return cleaned


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Detect dividers in audio files")
    parser.add_argument("audio", type=Path, help="Path to the audio file")
    parser.add_argument("markers", type=Path, help="Folder containing marker audio files")
    parser.add_argument("--threshold", type=float, default=0.85, help="Correlation threshold")
    parser.add_argument("--min-gap", type=float, default=8.0, help="Minimum gap between dividers in seconds")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    args: argparse.Namespace = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s %(name)s %(message)s")

    try:
        divider_paths: Dict[str, Path] = {}

        for marker in sorted(args.markers.glob("*")):
            name: str = marker.stem
            safe_name: str = "".join(c for c in name if c.isalnum() or c in "_ -")
            if safe_name:
                divider_paths[safe_name] = marker

        if not divider_paths:
            parser.error(f"No audio files found in {args.markers}")

        log.debug(f"Loading audio: {args.audio}")
        log.debug(f"Dividers to detect: {', '.join(divider_paths.keys())}")

        dividers: List[Tuple[float, str, float]] = detect_dividers(
            args.audio, divider_paths, args.threshold, args.min_gap
        )

        print(f"Found {len(dividers)} dividers:")
        for time, name, conf in dividers:
            mins: int = int(time // 60)
            secs: int = int(time % 60)
            print(f"  {mins:02d}:{secs:02d} ({time:.2f}s) - {name} - {conf * 100:.0f}%")
    except ValueError as e:
        log.error(f"Validation error: {e}")
        exit(1)
    except Exception as e:
        log.exception(f"Error: {e}")
        exit(1)
