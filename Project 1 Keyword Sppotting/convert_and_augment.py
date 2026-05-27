import os
import subprocess
import numpy as np
import librosa
import soundfile as sf

FFMPEG   = r"C:\Users\hasna\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe"
SRC_DIR  = r"c:\Users\hasna\Downloads\london-waves\london-origional"
OUT_DIR  = r"c:\Users\hasna\Downloads\london-waves\london-wavs"

os.makedirs(OUT_DIR, exist_ok=True)

def add_reverb(y, sr, delay_sec=0.05, decay=0.4, n_echoes=6):
    delay_samples = int(sr * delay_sec)
    out = y.copy().astype(np.float32)
    for i in range(1, n_echoes + 1):
        d = delay_samples * i
        if d >= len(y):
            break
        echo = np.zeros(len(y), dtype=np.float32)
        echo[d:] = y[:-d] * (decay ** i)
        out += echo
    return np.clip(out, -1.0, 1.0)

mp3_files = [f for f in os.listdir(SRC_DIR) if f.lower().endswith(".mp3")]
print(f"Found {len(mp3_files)} MP3 files. Converting + augmenting each...\n")

for fname in mp3_files:
    src_mp3 = os.path.join(SRC_DIR, fname)
    stem    = os.path.splitext(fname)[0]
    wav_out = os.path.join(OUT_DIR, stem + ".wav")

    # --- 1. Convert MP3 → WAV ---
    if not os.path.exists(wav_out):
        subprocess.run([FFMPEG, "-y", "-i", src_mp3, wav_out],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # --- 2. Load WAV for augmentation ---
    y, sr = librosa.load(wav_out, sr=None, mono=True)

    augmentations = {
        "pitch_up":    librosa.effects.pitch_shift(y, sr=sr, n_steps=2),
        "pitch_down":  librosa.effects.pitch_shift(y, sr=sr, n_steps=-2),
        "speed_up":    librosa.effects.time_stretch(y, rate=1.15),
        "speed_down":  librosa.effects.time_stretch(y, rate=0.85),
        "noise":       (y + 0.004 * np.random.randn(len(y))).astype(np.float32),
        "vol_up":      np.clip(y * 1.4, -1.0, 1.0),
        "vol_down":    y * 0.45,
        "reverb":      add_reverb(y, sr),
        "pitch_speed": librosa.effects.time_stretch(
                           librosa.effects.pitch_shift(y, sr=sr, n_steps=1),
                           rate=1.1),
    }

    for tag, aug_y in augmentations.items():
        out_path = os.path.join(OUT_DIR, f"{stem}__{tag}.wav")
        if not os.path.exists(out_path):
            sf.write(out_path, aug_y.astype(np.float32), sr)

    print(f"  Done: {stem}")

wav_count = len([f for f in os.listdir(OUT_DIR) if f.endswith(".wav")])
print(f"\nComplete! {wav_count} total WAV files in: {OUT_DIR}")
