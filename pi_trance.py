#!/usr/bin/env python3
"""
Pi Day Psy-Trance Generator
Generates a psy-trance beat (kick + bass) with spoken digits of Pi.
64 bars, one digit per beat, phrased in exhale/inhale groups. Output: MP3.
"""

import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
from pydub.effects import speedup
from gtts import gTTS
import os
import io

# === PARAMETERS ===
BPM = 140
SAMPLE_RATE = 44100
BARS = 64
BEATS_PER_BAR = 4
TOTAL_BEATS = BARS * BEATS_PER_BAR

# Phrasing: speak digits for PHRASE_SPEAK bars, rest for PHRASE_REST bars
PHRASE_SPEAK_BARS = 3   # 3 bars of digits (12 digits at 1/beat)
PHRASE_REST_BARS = 1    # 1 bar pause ("inhale")
PHRASE_BARS = PHRASE_SPEAK_BARS + PHRASE_REST_BARS  # 4-bar phrase cycle
NUM_PHRASES = BARS // PHRASE_BARS  # 16 phrases
DIGITS_PER_PHRASE = PHRASE_SPEAK_BARS * BEATS_PER_BAR  # 12 digits per phrase
TOTAL_DIGITS = NUM_PHRASES * DIGITS_PER_PHRASE  # 192 digits

# Pi digits — 192+ digits
PI_DIGITS = (
    "31415926535897932384626433832795028841971693993751"
    "05820974944592307816406286208998628034825342117067"
    "98214808651328230664709384460955058223172535940812"
    "84811174502841027019385211055596446229489549303819"
)
PI_DIGITS = PI_DIGITS[:TOTAL_DIGITS]

# Timing
beat_duration_sec = 60.0 / BPM
bar_duration_sec = beat_duration_sec * BEATS_PER_BAR
total_duration_sec = bar_duration_sec * BARS
total_samples = int(total_duration_sec * SAMPLE_RATE)


def generate_kick(duration_sec=0.15, sample_rate=SAMPLE_RATE):
    """Generate a psy-trance style kick drum with pitch sweep."""
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    # Frequency sweep from 150Hz down to 40Hz — punchy psy kick
    freq_start = 160
    freq_end = 38
    freq = freq_start * np.exp(-t * np.log(freq_start / freq_end) / duration_sec)
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate

    # Amplitude envelope — sharp attack, exponential decay
    envelope = np.exp(-t * 25)

    # Add initial click/transient
    click = np.exp(-t * 200) * 0.5

    kick = (np.sin(phase) * envelope + click * np.sin(2 * np.pi * 1000 * t)) * 0.8

    return kick


def generate_bassline(bar_index, bar_samples, sample_rate=SAMPLE_RATE):
    """Generate a psy-trance bassline pattern for one bar.
    Classic rolling 16th-note bass with slight pitch variation."""
    t = np.linspace(0, bar_samples / sample_rate, bar_samples, endpoint=False)

    # Base frequency — typical psy-trance bass around E1-A1
    # Slight variation per bar for movement
    base_freqs = [55, 55, 58.27, 55, 51.91, 55, 58.27, 61.74,
                  55, 55, 58.27, 55, 51.91, 55, 58.27, 55]
    base_freq = base_freqs[bar_index % len(base_freqs)]

    # 16th note pattern — the signature psy-trance rolling bass
    sixteenth = beat_duration_sec / 4
    bass = np.zeros(bar_samples)

    for i in range(16):  # 16 sixteenth notes per bar
        start = int(i * sixteenth * sample_rate)
        end = int((i + 1) * sixteenth * sample_rate)
        if end > bar_samples:
            end = bar_samples
        length = end - start
        if length <= 0:
            continue

        t_note = np.linspace(0, length / sample_rate, length, endpoint=False)

        # Slight pitch modulation on off-beats
        freq = base_freq * (1.02 if i % 2 == 1 else 1.0)

        # Square-ish wave with filtering effect (saw approximation)
        note = np.sin(2 * np.pi * freq * t_note)
        note += 0.5 * np.sin(2 * np.pi * freq * 2 * t_note)  # 2nd harmonic
        note += 0.25 * np.sin(2 * np.pi * freq * 3 * t_note)  # 3rd harmonic

        # Envelope per note — slight gate effect
        note_env = np.exp(-t_note * 15)
        note *= note_env

        bass[start:end] += note[:end - start]

    # Sidechain-style ducking on each beat
    for beat in range(BEATS_PER_BAR):
        duck_start = int(beat * beat_duration_sec * sample_rate)
        duck_len = int(0.08 * sample_rate)  # 80ms duck
        duck_end = min(duck_start + duck_len, bar_samples)
        duck_samples = duck_end - duck_start
        if duck_samples > 0:
            duck_curve = np.linspace(0.1, 1.0, duck_samples)
            bass[duck_start:duck_end] *= duck_curve

    return bass * 0.35


def generate_hihat(duration_sec=0.05, sample_rate=SAMPLE_RATE):
    """Generate a simple hihat sound using filtered noise."""
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    noise = np.random.randn(n_samples)
    envelope = np.exp(-t * 80)
    hihat = noise * envelope

    return hihat * 0.15


def generate_tts_digits():
    """Generate all 10 digit TTS samples, normalized to equal duration with same pitch."""
    digit_words = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three',
        '4': 'four', '5': 'five', '6': 'six', '7': 'seven',
        '8': 'eight', '9': 'nine'
    }

    # Step 1: Generate raw TTS for all 10 digits
    raw_segments = {}
    for d, word in digit_words.items():
        tts = gTTS(text=word, lang='en', slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        audio = AudioSegment.from_mp3(buf)
        audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
        # Strip silence from start and end
        audio = _strip_silence(audio)
        raw_segments[d] = audio
        print(f"    TTS '{word}': {len(audio)}ms")

    # Step 2: Target duration — fits within one beat with room to breathe
    # At 140 BPM, one beat = ~428ms. Target ~250ms for snappy delivery.
    target_ms = 250
    print(f"    Target duration: {target_ms}ms")

    # Step 3: Speed up longer samples (preserving pitch) to match target
    normalized = {}
    for d, seg in raw_segments.items():
        if len(seg) > target_ms + 30:  # needs speedup
            ratio = len(seg) / target_ms
            seg = speedup(seg, playback_speed=ratio, chunk_size=50, crossfade=25)
        # Hard-trim to exact target + apply fade out
        seg = seg[:target_ms]
        seg = seg.fade_out(30)
        # Pad if slightly shorter
        if len(seg) < target_ms:
            seg = seg + AudioSegment.silent(duration=target_ms - len(seg),
                                            frame_rate=SAMPLE_RATE)
        # Convert to numpy
        samples = np.array(seg.get_array_of_samples(), dtype=np.float64)
        samples = samples / 32768.0
        normalized[d] = samples

    return normalized


def _strip_silence(audio, silence_thresh=-40, chunk_size=10):
    """Strip leading and trailing silence from an AudioSegment."""
    # Find start
    start = 0
    for i in range(0, len(audio), chunk_size):
        if audio[i:i + chunk_size].dBFS > silence_thresh:
            start = max(0, i - chunk_size)
            break
    # Find end
    end = len(audio)
    for i in range(len(audio) - chunk_size, 0, -chunk_size):
        if audio[i:i + chunk_size].dBFS > silence_thresh:
            end = min(len(audio), i + 2 * chunk_size)
            break
    return audio[start:end]


def main():
    print("=== Pi Day Psy-Trance Generator ===")
    print(f"BPM: {BPM}, Bars: {BARS}, Digits: {PI_DIGITS}")
    print(f"Duration: {total_duration_sec:.1f} seconds")
    print()

    # Pre-generate sounds
    kick = generate_kick()
    hihat = generate_hihat()

    # Main mix buffer
    mix = np.zeros(total_samples)

    bar_samples = int(bar_duration_sec * SAMPLE_RATE)

    # === KICK on every beat ===
    print("Generating kick pattern...")
    for beat in range(TOTAL_BEATS):
        pos = int(beat * beat_duration_sec * SAMPLE_RATE)
        end = min(pos + len(kick), total_samples)
        mix[pos:end] += kick[:end - pos]

    # === HIHAT on off-beats (8th notes between kicks) ===
    print("Generating hihat pattern...")
    for eighth in range(TOTAL_BEATS * 2):
        if eighth % 2 == 1:  # off-beat 8th notes
            pos = int(eighth * beat_duration_sec * SAMPLE_RATE / 2)
            end = min(pos + len(hihat), total_samples)
            mix[pos:end] += hihat[:end - pos]

    # === BASSLINE per bar ===
    print("Generating bassline...")
    for bar in range(BARS):
        bar_start = int(bar * bar_duration_sec * SAMPLE_RATE)
        bar_end = min(bar_start + bar_samples, total_samples)
        actual_bar_samples = bar_end - bar_start
        bass = generate_bassline(bar, actual_bar_samples)
        mix[bar_start:bar_end] += bass[:actual_bar_samples]

    # === TTS Pi digits — one per beat, with phrase breathing ===
    # Pattern: 3 bars speak (12 digits) + 1 bar rest, in 4-bar phrases
    # Volume envelope per phrase simulates exhale (loud→fade) then silence (inhale)
    print("Generating Pi digit speech (phrased)...")
    tts_digits = generate_tts_digits()

    digit_index = 0
    for phrase in range(NUM_PHRASES):
        phrase_start_sec = phrase * PHRASE_BARS * bar_duration_sec

        for d in range(DIGITS_PER_PHRASE):
            if digit_index >= len(PI_DIGITS):
                break
            digit = PI_DIGITS[digit_index]
            speech = tts_digits[digit].copy()

            # Position within phrase: one digit per beat
            digit_time = phrase_start_sec + d * beat_duration_sec
            pos = int(digit_time * SAMPLE_RATE + beat_duration_sec * SAMPLE_RATE * 0.15)

            # Exhale envelope: starts strong, gently fades toward end of phrase
            # d goes 0..11, progress 0.0..1.0
            progress = d / max(DIGITS_PER_PHRASE - 1, 1)
            # Slight ramp up at start (first 2 digits), then gradual fade
            if progress < 0.15:
                vol = 0.6 + progress * 2.5  # ramp in
            else:
                vol = 1.0 - progress * 0.35  # gentle fade out (exhale running out)
            vol = max(vol, 0.4)

            end = min(pos + len(speech), total_samples)
            actual_len = end - pos
            if actual_len > 0:
                mix[pos:pos + actual_len] += speech[:actual_len] * 0.9 * vol

            digit_index += 1

    print(f"    Placed {digit_index} digits in {NUM_PHRASES} phrases")

    # === Normalize ===
    print("Mixing and normalizing...")
    peak = np.max(np.abs(mix))
    if peak > 0:
        mix = mix / peak * 0.95

    # === Export ===
    # Convert to 16-bit PCM
    mix_16bit = (mix * 32767).astype(np.int16)

    # Save as WAV first
    wav_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pi_trance.wav")
    wavfile.write(wav_path, SAMPLE_RATE, mix_16bit)

    # Convert to MP3
    mp3_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pi_trance.mp3")
    audio = AudioSegment.from_wav(wav_path)
    audio.export(mp3_path, format="mp3", bitrate="192k",
                 tags={"title": "Pi Trance", "artist": "Pi Day Generator",
                       "album": "3.14159..."})

    # Clean up WAV
    os.remove(wav_path)

    print()
    print(f"Done! Output: {mp3_path}")
    print(f"Duration: {total_duration_sec:.1f}s | BPM: {BPM} | Digits: {PI_DIGITS}")


if __name__ == "__main__":
    main()
