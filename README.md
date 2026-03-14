# π Trance

Psy-trance music generator driven by the digits of Pi.

Every digit of π controls the melody — spoken aloud over a full psy-trance production with kick, bassline, hihats, pads, and risers.

## 🎧 Listen in Browser

**[Open Pi Trance Web App](https://serg123e.github.io/Piday/)** — infinite real-time generation, no install needed.

Features:
- Toggle channels on/off: kick, bass, hihat, pad, riser, voice
- Adjust BPM, digits per phrase, rest beats, filter sweep, voice volume
- Scrolling digit display with current position highlighted

## 🎵 Generate Offline

### Prerequisites

```bash
pip install numpy scipy pydub gtts
```

FFmpeg is required for MP3 export:
```bash
# Ubuntu/Debian
sudo apt install ffmpeg
```

### Quick Track (~2 min)

```bash
python pi_trance.py
# Output: pi_trance.mp3 (64 bars, 76 digits, ~110 seconds)
```

### 1-Hour Track

```bash
python pi_trance_1h.py
# Output: pi_trance_1h.mp3 (2100 bars, 2620 digits, 60 minutes, ~82 MB)
```

## How It Works

### Structure

Digits of π are grouped into phrases of 5, separated by breathing pauses — like someone slowly reading π aloud over a trance beat.

```
Phrase 1:  "three"              (integer part, solo)
Phrase 2:  "one four one five nine"
Phrase 3:  "two six five three five"
...
```

Each phrase: 10 beats speaking + 6 beats rest = 16 beats (4 bars) at 140 BPM.

### Layers

| Layer | Description |
|-------|------------|
| **Kick** | Pitch-sweep sine (160→38 Hz) + click transient, every beat |
| **Bass** | 5-harmonic additive synth, 16th notes, sidechain ducking, evolving LP filter (300→1800 Hz) |
| **Hihat** | Filtered noise burst; pattern evolves from simple off-beat 8ths to complex 16th patterns |
| **Pad** | Detuned oscillators + LFO, plays during rest periods; alternates Em/Am chords |
| **Riser** | Noise with rising filter sweep, builds tension before each phrase |
| **Voice** | Google TTS digits, stress-aligned to 80ms, normalized to 400ms, exhale envelope per phrase |

### Tonality

Alternates between E minor and A minor every phrase:
- **E minor**: E1 (41.2 Hz) root, with G, A, B
- **A minor**: A1 (55 Hz) root, with C, D, E

## Files

| File | Description |
|------|------------|
| `index.html` | Self-contained web app (488 KB, includes base64 TTS samples) |
| `pi_trance.py` | Quick track generator (~110s) |
| `pi_trance_1h.py` | 1-hour track generator |
| `export_samples.py` | Exports stress-aligned digit samples as base64 WAV |

## License

MIT
