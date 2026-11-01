import argparse
from pathlib import Path

import whisper_transcribe as wt


def test_format_timestamp_rounds_and_carries():
    assert wt.format_timestamp(0) == "00:00:00,000"
    assert wt.format_timestamp(1.001) == "00:00:01,001"
    assert wt.format_timestamp(1.0004) == "00:00:01,000"
    # rounding must carry into the next minute / hour
    assert wt.format_timestamp(59.9996) == "00:01:00,000"
    assert wt.format_timestamp(3599.9996) == "01:00:00,000"
    assert wt.format_timestamp(61.234) == "00:01:01,234"
    assert wt.format_timestamp(3600.5) == "01:00:00,500"
    # invalid/negative timestamps are clamped instead of producing "-1:59,500"
    assert wt.format_timestamp(-0.5) == "00:00:00,000"


def test_format_timestamp_never_off_by_one():
    for ms in range(4000):
        seconds = ms / 1000 + 1.0
        assert wt.format_timestamp(seconds).split(",")[1] == f"{ms % 1000:03d}"


def test_normalize_text_collapses_newlines():
    assert wt.normalize_text(" hello \n world ") == "hello world"
    assert wt.normalize_text("\n") == ""


def test_find_media_files_is_case_insensitive_and_sorted(tmp_path):
    for name in ("b.mp3", "a.Mp3", "c.MP3", "d.wav", "notes.txt", "sub.srt"):
        (tmp_path / name).touch()
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "e.m4a").touch()

    found = wt.find_media_files(tmp_path)
    names = [p.name for p in found]
    assert names == sorted(names)
    assert set(names) == {"a.Mp3", "b.mp3", "c.MP3", "d.wav", "e.m4a"}
    assert all(p.suffix.lower() in wt.SUPPORTED_FORMATS for p in found)


def test_save_as_srt_skips_and_renumbers(tmp_path):
    result = {
        "segments": [
            {"start": 0.0, "end": 1.5, "text": " hello\nworld", "no_speech_prob": 0.1},
            {"start": 2.0, "end": 3.0, "text": "   ", "no_speech_prob": 0.9},
            {"start": 4.0, "end": 5.25, "text": "silence", "no_speech_prob": 0.9},
            {"start": 6.0, "end": 6.0004, "text": "kept"},
        ]
    }
    out = tmp_path / "out.srt"
    assert wt.save_as_srt(result, out, no_speech_threshold=0.6) == 2
    assert out.read_text(encoding="utf-8") == (
        "1\n00:00:00,000 --> 00:00:01,500\nhello world\n\n"
        "2\n00:00:06,000 --> 00:00:06,000\nkept\n\n"
    )
    # without filtering, the no-speech segment is still written
    assert wt.save_as_srt(result, out) == 3


def test_output_path_for():
    media = Path("recordings/a.mp3")
    assert wt.output_path_for(media, ".srt", None) == Path("recordings/a.srt")
    assert wt.output_path_for(media, ".srt", Path("out")) == Path("out/a.srt")


def test_save_outputs_warns_on_stem_collision(tmp_path, capsys):
    args = argparse.Namespace(srt=True, txt=False, output_dir=None, no_speech_threshold=1.0)
    result = {"text": "hi", "segments": [{"start": 0.0, "end": 1.0, "text": "hi"}]}
    # a.mp3 and a.wav share the same output name when written next to the inputs
    seen = set()
    wt.save_outputs(result, tmp_path / "a.mp3", args, seen)
    wt.save_outputs(result, tmp_path / "a.wav", args, seen)
    assert "Warning" in capsys.readouterr().out
    assert (tmp_path / "a.srt").read_text(encoding="utf-8").startswith("1\n")


def test_check_prerequisites(tmp_path):
    assert wt.check_prerequisites(tmp_path, None) == (
        "ffmpeg was not found on PATH (required by whisper.audio.load_audio)"
    )
    missing = tmp_path / "does-not-exist"
    assert wt.check_prerequisites(missing, "/usr/bin/ffmpeg") == f"input does not exist: {missing}"
    assert wt.check_prerequisites(tmp_path, "/usr/bin/ffmpeg") is None


def test_parser_defaults():
    args = wt.build_parser().parse_args(["sample.wav", "--srt"])
    assert args.model == wt.DEFAULT_MODEL
    assert args.device == wt.DEFAULT_DEVICE
    assert args.srt is True
    assert args.language is None
    assert wt.resolve_device("cpu") == "cpu"
    assert wt.resolve_device("auto") in {"cpu", "cuda", "mps"}
