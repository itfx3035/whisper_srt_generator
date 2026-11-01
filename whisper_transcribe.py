#!/usr/bin/env python3
"""Transcribe audio/video files with OpenAI Whisper and optionally write SRT subtitles.

Whisper and torch are imported lazily so the pure helpers (format_timestamp,
save_as_srt, find_media_files, ...) can be imported and tested without the heavy
ML dependencies installed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SUPPORTED_FORMATS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".webm",
    ".mkv",
    ".mp4",
    ".avi",
    ".mov",
}

DEFAULT_MODEL = "tiny"
DEFAULT_DEVICE = "cpu"


def import_whisper():
    """Import openai-whisper with a friendly error message."""
    try:
        import whisper
    except ImportError as exc:
        raise SystemExit(
            "openai-whisper is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return whisper


def available_model_choices():
    """Model names for argparse choices, or None when whisper is not installed."""
    try:
        import whisper
    except ImportError:
        return None
    return whisper.available_models()


def resolve_device(device: str) -> str:
    """Resolve 'auto' to the best available device; other values pass through."""
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm).

    Works in integer milliseconds so float truncation cannot produce
    off-by-one values, and carries correctly across minutes/hours.
    """
    total_ms = int(round(max(float(seconds), 0.0) * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def normalize_text(text: str) -> str:
    """Collapse internal newlines/whitespace so each SRT block is a single line."""
    return " ".join(text.split())


def find_media_files(directory: Path) -> List[Path]:
    """Return media files under `directory`, deterministically sorted.

    Matching is done on the lowercased suffix, so mixed-case extensions
    (e.g. `concert.Mp3`) are not skipped.
    """
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_FORMATS
    )


def output_path_for(media_file: Path, suffix: str, output_dir: Optional[Path]) -> Path:
    """Output path for a transcript/subtitle file, honouring `--output-dir`."""
    if output_dir is None:
        return media_file.with_suffix(suffix)
    return output_dir / f"{media_file.stem}{suffix}"


def save_as_srt(
    result: Dict[str, Any],
    output_path: Path,
    no_speech_threshold: float = 1.0,
) -> int:
    """Save a Whisper result as an SRT subtitle file. Returns number of blocks.

    Segments with no text, or with `no_speech_prob` above `no_speech_threshold`,
    are skipped and the remaining blocks are renumbered.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    # newline="\n" keeps output deterministic (SRT is conventionally LF).
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        for segment in result.get("segments") or []:
            text = normalize_text(segment.get("text", ""))
            if not text:
                continue
            if segment.get("no_speech_prob", 0.0) > no_speech_threshold:
                continue
            written += 1
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            f.write(f"{written}\n{start} --> {end}\n{text}\n\n")
    return written


def write_transcript(text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text.rstrip())
        f.write("\n")


def save_outputs(
    result: Dict[str, Any],
    media_file: Path,
    args: argparse.Namespace,
    seen_outputs: set,
) -> None:
    output_dir = Path(args.output_dir) if args.output_dir else None
    for suffix, writer in (
        (".txt", lambda p: write_transcript(result.get("text", ""), p)),
        (".srt", lambda p: save_as_srt(result, p, args.no_speech_threshold)),
    ):
        if not (args.txt if suffix == ".txt" else args.srt):
            continue
        path = output_path_for(media_file, suffix, output_dir)
        if path in seen_outputs:
            print(f"  Warning: {path} already written by another input file, overwriting")
        seen_outputs.add(path)
        writer(path)
        print(f"  Saved {suffix} to: {path}")


def transcribe_file(model, input_file: Path, args: argparse.Namespace) -> Dict[str, Any]:
    return model.transcribe(
        str(input_file),
        language=args.language,
        verbose=not args.quiet,
    )


def batch_transcribe(model, directory: Path, args: argparse.Namespace) -> int:
    """Transcribe all supported media files under `directory`. Returns failures."""
    media_files = find_media_files(directory)
    if not media_files:
        print("No media files found in directory.")
        return 0

    print(f"Found {len(media_files)} media file(s) to process")
    seen_outputs: set = set()
    failures = 0

    for i, media_file in enumerate(media_files, 1):
        print(f"\n[{i}/{len(media_files)}] Processing: {media_file}")
        started = time.monotonic()
        try:
            result = transcribe_file(model, media_file, args)
            save_outputs(result, media_file, args, seen_outputs)
            if args.print_text:
                print(f"  {result.get('text', '').strip()}")
            print(f"  Done in {time.monotonic() - started:.1f}s")
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - keep processing the rest of the batch
            print(f"  Error: {exc}", file=sys.stderr)
            failures += 1

    if failures:
        print(f"\nCompleted with {failures} failure(s).", file=sys.stderr)
    return failures


def check_prerequisites(input_path: Path, ffmpeg_path: Optional[str]) -> Optional[str]:
    """Return an error message if the run cannot succeed, otherwise None."""
    if ffmpeg_path is None:
        return "ffmpeg was not found on PATH (required by whisper.audio.load_audio)"
    if not input_path.exists():
        return f"input does not exist: {input_path}"
    return None


def build_parser(model_choices: Optional[List[str]] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe audio/video files using Whisper and write SRT subtitles",
    )
    parser.add_argument("input", help="Audio/video file or directory to process")
    model_options: Dict[str, Any] = {
        "default": DEFAULT_MODEL,
        "help": f"Whisper model name (default: {DEFAULT_MODEL})",
    }
    if model_choices:
        model_options["choices"] = model_choices
    parser.add_argument("--model", "-m", **model_options)
    parser.add_argument(
        "--language",
        "-l",
        default=None,
        help="Spoken language code (e.g., 'en', 'pl'). Auto-detected if omitted",
    )
    parser.add_argument(
        "--srt",
        "-s",
        action="store_true",
        help="Save transcription as an SRT subtitle file",
    )
    parser.add_argument(
        "--txt",
        action="store_true",
        help="Save the full transcription as a .txt file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Directory for generated .txt/.srt files (default: alongside the input)",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        choices=["cpu", "cuda", "mps", "auto"],
        help=f"PyTorch device (default: {DEFAULT_DEVICE}; 'auto' picks the best available)",
    )
    parser.add_argument(
        "--no-speech-threshold",
        type=float,
        default=1.0,
        help="Skip segments whose no_speech_prob exceeds this value (1.0 disables filtering)",
    )
    parser.add_argument(
        "--print-text",
        action="store_true",
        help="Print each transcript in batch mode",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress Whisper per-segment progress output",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser(model_choices=available_model_choices()).parse_args(argv)

    input_path = Path(args.input)
    error = check_prerequisites(input_path, shutil.which("ffmpeg"))
    if error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    whisper = import_whisper()
    device = resolve_device(args.device)
    print(f"Loading Whisper model '{args.model}' on {device}...")
    # Load directly on the target device instead of moving the model afterwards.
    model = whisper.load_model(args.model, device=device)

    try:
        if input_path.is_dir():
            failures = batch_transcribe(model, input_path, args)
            return 1 if failures else 0

        print(f"Transcribing: {input_path}")
        result = transcribe_file(model, input_path, args)

        if not args.quiet:
            print("\n=== Transcription Result ===")
            print(result.get("text", ""))

        if args.txt or args.srt:
            save_outputs(result, input_path, args, set())
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
