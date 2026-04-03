# Whisper Audio Transcription (Native Installation)

Transcribe audio files using OpenAI's Whisper AI model, optimized for CPU inference.

## Features

- Supports multiple audio formats: WAV, MP3, M4A, FLAC, OGG, WEBM, MKV, MP4, AVI, MOV
- 5 model sizes available (tiny → large)
- Auto-detects language or specify explicitly
- Single file or batch directory processing
- SRT subtitle file output option

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
cd whisper
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

## Command Options

| Flag | Description | Default |
|------|-------------|---------|
| `input` | Audio file path or directory to process | required |
| `-m`, `--model` | Model size: tiny, base, small, medium, large | tiny |
| `-l`, `--language` | Language code (en, pl, de, fr, etc.) | auto-detect |
| `-s`, `--srt` | Save transcription as SRT subtitle file | disabled |

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

- Transcription printed to stdout
- In batch mode, `.txt` files saved alongside audio files
- With `--srt` flag, `.srt` subtitle files are created

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
- Ensure FFmpeg is in your system PATH

**Error: "No audio files found"**
- Ensure the directory contains audio files with supported extensions
- Check file extensions are lowercase or uppercase (both supported)

**Slow processing?**
- Use `--model tiny` or `--model base` for faster results
- Larger models provide better accuracy but take longer

**Memory issues?**
- Use smaller model (tiny/base)
- Process files individually instead of batch mode

**First run is very slow?**
- Normal behavior - Whisper model is being downloaded
- Subsequent runs will be faster (model is cached)

## License

OpenAI Whisper is released under MIT License.
