#!/usr/bin/env python3
"""
Pi Day Psy-Trance Generator
Generates a psy-trance beat (kick + bass) with spoken digits of Pi.
64 bars, one digit every 2 beats. Output: MP3.
"""

import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
from gtts import gTTS
import tempfile
import os
import struct
import io

# === PARAMETERS ===
BPM = 140
SAMPLE_RATE = 44100
BARS = 64
BEATS_PER_BAR = 4
TOTAL_BEATS = BARS * BEATS_PER_BAR
DIGITS_PER_BAR = 2  # one digit every 2 beats

# Pi digits: 3.14159265358979323846...
# 64 bars x 2 digits per bar = 128 digits
PI_DIGITS = "31415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"
PI_DIGITS += "82148086513282306647093844609550582231725359"
PI_DIGITS = PI_DIGITS[:BARS * DIGITS_PER_BAR]  # 128 digits

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


def text_to_speech_digit(digit_char):
    """Generate TTS audio for a single Pi digit in English."""
    digit_words = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three',
        '4': 'four', '5': 'five', '6': 'six', '7': 'seven',
        '8': 'eight', '9': 'nine'
    }
    word = digit_words.get(digit_char, digit_char)

    tts = gTTS(text=word, lang='en', slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)

    audio = AudioSegment.from_mp3(buf)
    audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)

    # Convert to numpy
    samples = np.array(audio.get_array_of_samples(), dtype=np.float64)
    samples = samples / 32768.0  # normalize to [-1, 1]

    return samples


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

    # === TTS Pi digits — one every 2 beats ===
    print("Generating Pi digit speech...")
    half_bar_sec = beat_duration_sec * 2  # interval between digits
    # Cache TTS to avoid re-generating the same digit
    tts_cache = {}
    for i, digit in enumerate(PI_DIGITS):
        if i % 16 == 0:
            print(f"  Digits {i + 1}-{min(i + 16, len(PI_DIGITS))} of {len(PI_DIGITS)}...")

        if digit not in tts_cache:
            tts_cache[digit] = text_to_speech_digit(digit)
        speech = tts_cache[digit].copy()

        # Position: every 2 beats, offset by half a beat
        pos = int(i * half_bar_sec * SAMPLE_RATE + beat_duration_sec * SAMPLE_RATE * 0.3)

        # Trim if speech is too long for the slot
        max_len = int(half_bar_sec * SAMPLE_RATE * 0.85)
        if len(speech) > max_len:
            speech = speech[:max_len]

        end = min(pos + len(speech), total_samples)
        actual_len = end - pos
        if actual_len > 0:
            mix[pos:pos + actual_len] += speech[:actual_len] * 0.9

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
