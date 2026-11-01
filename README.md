# Whisper Audio Transcription (Native Installation)

Transcribe audio files using OpenAI's Whisper AI model, optimized for CPU inference.

## Features

- Supports multiple audio formats: WAV, MP3, M4A, FLAC, OGG, WEBM, MKV, MP4, AVI, MOV
- 5 model sizes available (tiny → large), plus `large-v3` and the `.en`/`distil-*` variants
- Auto-detects language or specify explicitly
- Single file or batch directory processing
- SRT subtitle file output option
- CPU inference by default, with `--device cuda/mps/auto` when a GPU is available

## Prerequisites

- Python 3.8 or higher
- FFmpeg (required for audio processing)

### Installing FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Debian/Ubuntu):**
```bash
sudo apt-get update && sudo apt-get install ffmpeg
```

**Linux (Fedora/RHEL):**
```bash
sudo dnf install ffmpeg
```

**Windows:**
- Download from [ffmpeg.org](https://ffmpeg.org/download.html) or
- Use Chocolatey: `choco install ffmpeg`

## Installation

### 1. Clone or navigate to the project directory

```bash
cd whisper_srt_generator
```

### 2. Create a virtual environment (recommended)

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\Activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** First run will download the Whisper model automatically (may take a few minutes)

`torchaudio` is not required — Whisper decodes audio by calling the `ffmpeg` command line tool.

### Running the tests

```bash
pip install pytest
python3 -m pytest
```

The pure helpers (`format_timestamp`, `save_as_srt`, `find_media_files`, `check_prerequisites`) can be
imported without Whisper installed, so the suite runs on a bare environment.

## Usage

### Single File Transcription

**Basic usage (auto-detect language):**
```bash
python whisper_transcribe.py audio.mp3
```

**Specify model size:**
```bash
python whisper_transcribe.py audio.wav --model small
```

**Specify language:**
```bash
python whisper_transcribe.py audio.wav --language en
```

**Save as SRT subtitle file:**
```bash
python whisper_transcribe.py video.mp4 --srt
```

**Combine options:**
```bash
python whisper_transcribe.py audio.m4a --model base --language pl --srt
```

### Batch Directory Processing

Process all audio/video files in a directory:

```bash
python whisper_transcribe.py ./recordings --model base
```

With SRT output:
```bash
python whisper_transcribe.py ./recordings --model tiny --srt
```

Write results to a separate directory (avoids overwriting existing `.txt`/`.srt` files that share a
stem with another input, e.g. `a.mp3` and `a.wav`), and skip hallucinated segments on silence:

```bash
python whisper_transcribe.py ./recordings --srt --txt -o ./out --no-speech-threshold 0.6
```

Use a GPU when one is available:

```bash
python whisper_transcribe.py audio.mp3 --device auto
```

## Command Options

| Flag | Description | Default |
|------|-------------|---------|
| `input` | Audio file path or directory to process | required |
| `-m`, `--model` | Model name (any `whisper.available_models()` value) | `tiny` |
| `-l`, `--language` | Language code (en, pl, de, fr, etc.) | auto-detect |
| `-s`, `--srt` | Save transcription as SRT subtitle file | disabled |
| `--txt` | Save the full transcript as a `.txt` file | disabled |
| `-o`, `--output-dir` | Where generated files are written | alongside the input |
| `--device` | `cpu`, `cuda`, `mps`, `auto` | `cpu` |
| `--no-speech-threshold` | Drop segments whose `no_speech_prob` is higher (1.0 = disabled) | `1.0` |
| `--print-text` | Print each transcript in batch mode | disabled |
| `-q`, `--quiet` | Suppress Whisper per-segment progress | disabled |

## Model Sizes Comparison

| Model | Size | RAM Required | Speed (CPU) | Quality |
|-------|------|--------------|-------------|---------|
| tiny | 75 MB | ~1 GB | Fastest | Decent |
| base | 142 MB | ~2 GB | Fast | Good |
| small | 466 MB | ~3 GB | Medium | Very Good |
| medium | 1.5 GB | ~5 GB | Slow | Excellent |
| large | 3 GB | ~8 GB | Slowest | Best |

**Recommendation:** Start with `base` or `small` for best CPU performance.

## Output

- Transcription printed to stdout (single-file mode; `--print-text` for batch mode)
- `--txt` writes the full transcript and `--srt` writes subtitles — both work for single files and
  batch runs (batch mode used to always write `.txt`; that now requires `--txt`)
- Files are written next to the input unless `--output-dir` is given
- SRT blocks are single-line, renumbered after skipped segments, and timestamps are rounded to the
  nearest millisecond
- Exit code is `1` when any input fails or the input path is missing, `130` on Ctrl-C

## Example Outputs

```bash
python whisper_transcribe.py sample.wav --model tiny

# Output:
# Loading Whisper model 'tiny'...
# Transcribing: sample.wav
#
# === Transcription Result ===
# Hello, this is a test. This is just some audio.
```

With SRT output:
```bash
python whisper_transcribe.py video.mp4 --srt

# Creates: video.srt
# 1
# 00:00:00,000 --> 00:00:05,000
# Hello, this is a test.
```

## Troubleshooting

**Error: "ffmpeg not found"**
- Install FFmpeg (see Prerequisites section)
- Ensure FFmpeg is on your system PATH — the script checks for it before loading the model

**Error: "No media files found in directory"**
- Ensure the directory contains media files with supported extensions
- Extension matching is case-insensitive (`concert.Mp3` is included)
- A missing directory is reported as "input does not exist", not as an empty directory

**Slow processing?**
- Use `--model tiny` or `--model base` for faster results
- Larger models provide better accuracy but take longer

**Memory issues?**
- Use a smaller model (tiny/base or `distil-*`)
- Process files individually instead of batch mode

**First run is very slow?**
- Normal behavior - Whisper model is being downloaded
- Subsequent runs will be faster (model is cached)

## License

This project is released under the MIT License (see `LICENSE`). OpenAI Whisper, which it depends on,
is also MIT licensed.
