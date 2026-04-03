import whisper
import argparse
from pathlib import Path
import torch


def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format (HH:MM:SS,mmm).
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def save_as_srt(result: dict, output_path: str):
    """
    Save transcription result as SRT subtitle file.
    
    Args:
        result: Whisper transcription result with 'segments' key
        output_path: Path to save the SRT file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(result['segments'], 1):
            start = format_timestamp(segment['start'])
            end = format_timestamp(segment['end'])
            text = segment['text'].strip()
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")


def transcribe_audio(input_file: str, model_size: str = "tiny", language: str = None, output_srt: bool = False):
    """
    Transcribe audio file using Whisper model on CPU.
    
    Args:
        input_file: Path to audio/video file (supports wav, mp3, m4a, flac, mkv, mp4, webm, etc.)
        model_size: One of 'tiny', 'base', 'small', 'medium', 'large'
        language: Optional language code (e.g., 'en', 'pl', 'de')
        output_srt: If True, save transcription as SRT subtitle file
    """
    print(f"Loading Whisper model '{model_size}'...")
    model = whisper.load_model(model_size)
    
    # Set to CPU mode
    if torch is not None:
        model = model.to("cpu")
    
    print(f"Transcribing: {input_file}")
    result = model.transcribe(
        input_file,
        language=language,
        verbose=True
    )
    
    print("\n=== Transcription Result ===")
    print(result["text"])
    
    # Save as SRT if requested
    if output_srt:
        input_path = Path(input_file)
        srt_path = input_path.with_suffix('.srt')
        save_as_srt(result, str(srt_path))
        print(f"\nSRT subtitles saved to: {srt_path}")
    
    return result


def batch_transcribe(directory: str, model_size: str = "tiny", language: str = None, output_srt: bool = False):
    """
    Transcribe all audio/video files in a directory.
    
    Supported formats: .wav, .mp3, .m4a, .flac, .ogg, .webm, .mkv, .mp4, .avi, .mov
    """
    from pathlib import Path
    import os
    
    supported_formats = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm', '.mkv', '.mp4', '.avi', '.mov'}
    media_files = []
    
    for ext in supported_formats:
        media_files.extend(Path(directory).rglob(f'*{ext}'))
        media_files.extend(Path(directory).rglob(f'*{ext.upper()}'))
    
    if not media_files:
        print("No media files found in directory.")
        return
    
    print(f"Found {len(media_files)} media file(s) to process")
    
    # Load model once for all files
    model = whisper.load_model(model_size)
    model = model.to("cpu")
    
    results = {}
    for i, media_file in enumerate(media_files, 1):
        print(f"\n[{i}/{len(media_files)}] Processing: {media_file}")
        try:
            result = model.transcribe(str(media_file), language=language, verbose=False)
            results[str(media_file)] = result["text"]
            
            # Save transcript to file
            transcript_file = media_file.with_suffix('.txt')
            with open(transcript_file, 'w', encoding='utf-8') as f:
                f.write(result["text"])
            print(f"  Saved transcript to: {transcript_file}")
            
            # Save SRT if requested
            if output_srt:
                srt_path = media_file.with_suffix('.srt')
                save_as_srt(result, str(srt_path))
                print(f"  Saved SRT to: {srt_path}")
        except Exception as e:
            print(f"  Error: {e}")
            results[str(media_file)] = None
    
    return results


if __name__ == "__main__":
    import torch
    
    parser = argparse.ArgumentParser(description="Transcribe audio using Whisper (CPU)")
    parser.add_argument("input", help="Audio file or directory to process")
    parser.add_argument("--model", "-m", default="tiny", 
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size (default: tiny)")
    parser.add_argument("--language", "-l", default=None,
                       help="Spoken language code (e.g., 'en', 'pl'). Auto-detect if not specified")
    parser.add_argument("--srt", "-s", action="store_true",
                       help="Save transcription as SRT subtitle file")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_dir():
        batch_transcribe(str(input_path), args.model, args.language, output_srt=args.srt)
    else:
        transcribe_audio(str(input_path), args.model, args.language, output_srt=args.srt)
