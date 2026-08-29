"""Build 6 demo sample videos: real/fake visual x natural/TTS-style audio mixes.

Uses existing frame-derived visual tracks (data/demo) + speech WAVs (svara
CREMA-D natural speech) + synthetic-speech WAVs from the Kaggle audio subset
(real/fake clips there ARE the natural/TTS speech we need for audio forensics).

Output: app/samples/*.mp4 (small, H.264+AAC, ~6s)
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo"
AUD = ROOT / "data" / "audio_unidpro"
OUT = ROOT / "app" / "samples"
OUT.mkdir(parents=True, exist_ok=True)


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("FFMPEG FAIL:", " ".join(cmd)[:160])
        print(r.stderr[-600:])
        sys.exit(1)


def pick_audio(kind: str, n: int) -> Path:
    """kind 'natural' -> CREMA-D natural speech; 'tts' -> unidpro synthetic (voice-clone/TTS)."""
    if kind == "natural":
        cands = sorted((ROOT / "data" / "audio_cremad").glob("*.wav"))
    else:
        cands = sorted(AUD.glob("*synthetic*"))
    if not cands:
        sys.exit(f"no audio candidates for {kind}")
    return cands[n % len(cands)]


def audio_chain(src: Path) -> list:
    # 16k mono, normalize loudness, pad/trim to 6s
    return ["-i", str(src),
            "-filter_complex",
            f"[1:a]aresample=16000,pan=mono|c0=c0,loudnorm=I=-16:TP=-1.5:LRA=11,apad=whole_dur=6,atrim=0:6[a]"]


def make(visual: str, audio_kind: str, tag: str, a_idx: int):
    vid = DEMO / f"{visual}_sample.mp4"
    aud = pick_audio(audio_kind, a_idx)
    out = OUT / f"{tag}.mp4"
    # loop the short visual track to 3s so one full 2s audio segment fits
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-stream_loop", "2", "-i", str(vid)] + audio_chain(aud) + [
          "-map", "0:v", "-map", "[a]",
          "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
          "-c:a", "aac", "-b:a", "96k", "-shortest",
          str(out)]
    run(cmd)
    print(f"built {out.name}  ({visual} video + {audio_kind} audio)")


if __name__ == "__main__":
    make("real", "natural", "sample_real_video_natural_voice", 0)
    make("fake", "natural", "sample_fake_video_natural_voice", 1)
    make("real", "tts", "sample_real_video_synthetic_voice", 2)
    make("fake", "tts", "sample_fake_video_synthetic_voice", 3)
    # long-form: concatenate fake video visual track x2 for a 12s clip
    print("SAMPLES_DONE")
