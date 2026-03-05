# ChapteRSS

Automatically add chapter markers to podcast episodes by detecting audio dividers.

## Overview

ChapteRSS provides tools to:
1. **Extract clips** from audio files using SRT annotations
2. **Detect dividers** in podcast episodes using audio fingerprinting
3. **Insert chapters** into MP3 files based on detected dividers
4. **Serve an RSS feed** that automatically adds chapters to episodes

## Installation

```bash
uv sync
```

## Quick Start: Manual Workflow

### Step 1: Extract Divider Clips

First, you need sample audio clips of the dividers (jingles, music, sound effects) that separate sections in your podcast.

Create an SRT file marking where dividers appear in a sample episode:

```srt
1
00:00:05,000 --> 00:00:08,500
intro

2
00:15:30,200 --> 00:15:33,800
sponsor_break

3
00:30:45,100 --> 00:30:48,600
outro
```

Extract the clips:

```bash
uv run extract-clips episode.mp3 markers.srt -o dividers/
```

This creates:
- `dividers/intro.wav`
- `dividers/sponsor_break.wav`
- `dividers/outro.wav`

### Step 2: Detect Dividers

Test that the dividers are being detected correctly in your episodes:

```bash
uv run detect-dividers episode.mp3 dividers/
```

Output example:
```
Found 3 dividers:
  00:05 (5.23s) - intro - 89%
  15:30 (930.45s) - sponsor_break - 86%
  30:45 (1845.12s) - outro - 90%
```

#### Tuning Detection

If detection isn't working well, adjust these parameters:

```bash
# Lower threshold for more sensitive detection (more false positives)
uv run detect-dividers episode.mp3 dividers/ --threshold 0.75

# Higher threshold for stricter detection (may miss some dividers)
uv run detect-dividers episode.mp3 dividers/ --threshold 0.90

# Adjust minimum gap between detections (default 8 seconds)
uv run detect-dividers episode.mp3 dividers/ --min-gap 10.0
```

### Step 3: Insert Chapters

Once detection looks good, insert chapter markers into your MP3:

```bash
uv run insert-chapters episode.mp3 dividers/ episode_with_chapters.mp3
```

This creates a new MP3 file with embedded chapter markers. You can verify the chapters in most podcast players or with:

```bash
ffprobe -show_chapters episode_with_chapters.mp3
```

## Automatic Server Mode

For ongoing podcast processing, use the server to automatically add chapters to all episodes in an RSS feed.

### Server Configuration

Create a configuration directory structure:

```
config/
└── my_podcast/
    ├── config.yaml
    └── markers/
        ├── intro.wav
        ├── sponsor_break.wav
        └── outro.wav
```

**config/my_podcast/config.yaml:**
```yaml
source_rss: https://example.com/original-feed.xml
```

### Running the Server

Start the server:

```bash
uv run server --host 0.0.0.0 --port 8000
```

Or with auto-reload during development:

```bash
uv run server --reload
```

### Using the Server

1. **List available podcasts:**
   ```
   curl http://localhost:8000/list
   ```

2. **Access the modified RSS feed:**
   ```
   curl http://localhost:8000/rss/my_podcast
   ```

3. **Individual episodes (with chapters):**
   ```
   wget http://localhost:8000/audio/my_podcast/{episode_id}.mp3
   ```

The server will:
- Download the original RSS feed
- Cache it locally
- Generate a new RSS feed with modified enclosure URLs
- When an episode is requested:
  - Download the original audio (if not cached)
  - Detect dividers and add chapters
  - Cache the result
  - Serve the chaptered version

### Directory Structure

When running, the server creates:

```
data/
└── my_podcast/
    ├── original/
    │   ├── feed.xml           # Original RSS feed
    │   └── episode123.mp3     # Original episodes
    └── chapped/
        ├── feed.xml           # Modified RSS feed
        └── episode123.mp3     # Episodes with chapters
```
