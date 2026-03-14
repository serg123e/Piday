#!/usr/bin/env python3
"""
Pi Day Psy-Trance Generator
Generates a psy-trance beat (kick + bass) with spoken digits of Pi.
64 bars, groups of 5 decimal digits with breathing pauses.
Evolving trance effects: risers, filter sweeps, pads.
Output: MP3.
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter
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

# Phrasing: "3" alone in first phrase, then groups of 5 decimal digits
BEATS_PER_DIGIT = 2      # one digit every 2 beats
DIGITS_PER_GROUP = 5     # 5 digits per "exhale" group
SPEAK_BEATS = DIGITS_PER_GROUP * BEATS_PER_DIGIT  # 10 beats speaking
REST_BEATS = 6           # 6 beats pause ("inhale") → total 16 beats = 4 bars
PHRASE_BEATS = SPEAK_BEATS + REST_BEATS  # 16 beats = 4 bars per phrase
PHRASE_BARS = PHRASE_BEATS // BEATS_PER_BAR  # 4 bars
NUM_PHRASES = BARS // PHRASE_BARS  # 16 phrases

# First phrase: just "3" (the integer part), then 15 groups of 5 decimal digits = 75
TOTAL_DIGITS = 1 + (NUM_PHRASES - 1) * DIGITS_PER_GROUP  # 76

# Pi digits: "3" + decimal digits
PI_INTEGER = "3"
PI_DECIMALS = (
    "14159265358979323846264338327950288419716939937510"
    "58209749445923078164062862089986280348253421170679"
)

# Timing
beat_duration_sec = 60.0 / BPM
bar_duration_sec = beat_duration_sec * BEATS_PER_BAR
total_duration_sec = bar_duration_sec * BARS
total_samples = int(total_duration_sec * SAMPLE_RATE)


def generate_kick(duration_sec=0.15, sample_rate=SAMPLE_RATE):
    """Generate a psy-trance style kick drum with pitch sweep."""
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    freq_start = 160
    freq_end = 38
    freq = freq_start * np.exp(-t * np.log(freq_start / freq_end) / duration_sec)
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate

    envelope = np.exp(-t * 25)
    click = np.exp(-t * 200) * 0.5

    kick = (np.sin(phase) * envelope + click * np.sin(2 * np.pi * 1000 * t)) * 0.8
    return kick


def get_phrase_for_bar(bar_index):
    """Return phrase index (0-15) for a given bar."""
    return bar_index // PHRASE_BARS


def generate_bassline(bar_index, bar_samples, sample_rate=SAMPLE_RATE):
    """Generate a psy-trance bassline with evolving filter and phrase tonality."""
    phrase = get_phrase_for_bar(bar_index)
    bar_in_phrase = bar_index % PHRASE_BARS

    # Two tonalities alternating by phrase:
    # Odd phrases (1,3,5...): A minor — root A1=55Hz, with C, D, E
    # Even phrases (0,2,4...): E minor — root E1=41.2Hz, with G, A, B
    if phrase % 2 == 0:
        # E minor
        base_freqs = [41.20, 41.20, 49.00, 41.20, 46.25, 41.20, 49.00, 55.00]
    else:
        # A minor
        base_freqs = [55.00, 55.00, 58.27, 55.00, 51.91, 55.00, 58.27, 61.74]
    base_freq = base_freqs[bar_in_phrase % len(base_freqs)]

    sixteenth = beat_duration_sec / 4
    bass = np.zeros(bar_samples)

    for i in range(16):
        start = int(i * sixteenth * sample_rate)
        end = int((i + 1) * sixteenth * sample_rate)
        if end > bar_samples:
            end = bar_samples
        length = end - start
        if length <= 0:
            continue

        t_note = np.linspace(0, length / sample_rate, length, endpoint=False)
        freq = base_freq * (1.02 if i % 2 == 1 else 1.0)

        # Richer harmonics for filter sweep to work on
        note = np.sin(2 * np.pi * freq * t_note)
        note += 0.6 * np.sin(2 * np.pi * freq * 2 * t_note)
        note += 0.35 * np.sin(2 * np.pi * freq * 3 * t_note)
        note += 0.2 * np.sin(2 * np.pi * freq * 4 * t_note)
        note += 0.1 * np.sin(2 * np.pi * freq * 5 * t_note)

        note_env = np.exp(-t_note * 15)
        note *= note_env
        bass[start:end] += note[:end - start]

    # Sidechain ducking
    for beat in range(BEATS_PER_BAR):
        duck_start = int(beat * beat_duration_sec * sample_rate)
        duck_len = int(0.08 * sample_rate)
        duck_end = min(duck_start + duck_len, bar_samples)
        duck_samples = duck_end - duck_start
        if duck_samples > 0:
            duck_curve = np.linspace(0.1, 1.0, duck_samples)
            bass[duck_start:duck_end] *= duck_curve

    # Evolving low-pass filter: opens gradually over the track
    # bar_index 0→63 maps cutoff 300Hz→1800Hz
    progress = bar_index / max(BARS - 1, 1)
    cutoff = 300 + progress * 1500
    nyq = sample_rate / 2
    cutoff_norm = min(cutoff / nyq, 0.99)
    b, a = butter(2, cutoff_norm, btype='low')
    bass = lfilter(b, a, bass)

    return bass * 0.35


def generate_hihat(duration_sec=0.05, sample_rate=SAMPLE_RATE):
    """Generate a hihat sound."""
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    noise = np.random.randn(n_samples)
    envelope = np.exp(-t * 80)
    return noise * envelope * 0.15



def generate_riser(duration_sec, sample_rate=SAMPLE_RATE):
    """Generate a noise riser/sweep leading into the next phrase."""
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    noise = np.random.randn(n_samples)

    # Volume crescendo
    envelope = (t / duration_sec) ** 2

    # Rising filter sweep
    riser = np.zeros(n_samples)
    chunk_size = int(0.02 * sample_rate)  # 20ms chunks
    for i in range(0, n_samples - chunk_size, chunk_size):
        progress = i / n_samples
        cutoff = 500 + progress * 8000
        nyq = sample_rate / 2
        cutoff_norm = min(cutoff / nyq, 0.99)
        b, a = butter(2, cutoff_norm, btype='low')
        chunk = noise[i:i + chunk_size]
        riser[i:i + chunk_size] = lfilter(b, a, chunk)

    riser *= envelope
    return riser * 0.15


def generate_pad_note(duration_sec, freq, sample_rate=SAMPLE_RATE):
    """Generate a soft pad tone with slow attack/release."""
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    # Detuned saw-like pad (two slightly detuned oscillators)
    pad = np.sin(2 * np.pi * freq * t)
    pad += np.sin(2 * np.pi * freq * 1.003 * t)  # slight detune
    pad += 0.5 * np.sin(2 * np.pi * freq * 2.001 * t)  # octave

    # Slow attack and release envelope
    attack = int(0.3 * sample_rate)
    release = int(0.5 * sample_rate)
    envelope = np.ones(n_samples)
    if attack < n_samples:
        envelope[:attack] = np.linspace(0, 1, attack)
    if release < n_samples:
        envelope[-release:] = np.linspace(1, 0, release)

    # Gentle LFO for movement
    lfo = 1.0 + 0.15 * np.sin(2 * np.pi * 0.25 * t)

    pad *= envelope * lfo
    return pad * 0.06


def _find_stress_position(samples, sample_rate):
    """Find the stress (peak energy) position using RMS envelope."""
    window = int(0.020 * sample_rate)
    hop = window // 4
    rms = np.array([np.sqrt(np.mean(samples[i:i + window] ** 2))
                     for i in range(0, len(samples) - window, hop)])
    peak_idx = np.argmax(rms)
    return peak_idx * hop


def generate_tts_digits():
    """Generate all 10 digit TTS samples, stress-aligned and duration-normalized."""
    digit_words = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three',
        '4': 'four', '5': 'five', '6': 'six', '7': 'seven',
        '8': 'eight', '9': 'nine'
    }

    raw_data = {}
    for d, word in digit_words.items():
        tts = gTTS(text=word, lang='en', slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        audio = AudioSegment.from_mp3(buf)
        audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
        audio = _strip_silence(audio)

        samp = np.array(audio.get_array_of_samples(), dtype=np.float64) / 32768.0
        stress_sample = _find_stress_position(samp, SAMPLE_RATE)
        stress_ms = stress_sample / SAMPLE_RATE * 1000

        raw_data[d] = (audio, stress_ms)
        print(f"    TTS '{word}': {len(audio)}ms, stress at {stress_ms:.0f}ms")

    target_ms = 400
    sped_up = {}
    for d, (seg, stress_ms) in raw_data.items():
        if len(seg) > target_ms + 30:
            ratio = len(seg) / target_ms
            seg_fast = speedup(seg, playback_speed=ratio, chunk_size=50, crossfade=25)
            samp = np.array(seg_fast.get_array_of_samples(), dtype=np.float64) / 32768.0
            new_stress = _find_stress_position(samp, SAMPLE_RATE) / SAMPLE_RATE * 1000
        else:
            seg_fast = seg
            new_stress = stress_ms
        sped_up[d] = (seg_fast, new_stress)

    STRESS_TARGET_MS = 80
    stress_overrides = {'9': -70, '1': -40, '6': -30, '8': -10}
    print(f"    Aligning stress to {STRESS_TARGET_MS}ms, target total {target_ms}ms")

    normalized = {}
    for d, (seg, stress_ms) in sped_up.items():
        word = digit_words[d]
        override = stress_overrides.get(d, 0)
        shift_ms = int(STRESS_TARGET_MS + override - stress_ms)

        if shift_ms > 0:
            seg = AudioSegment.silent(duration=shift_ms, frame_rate=SAMPLE_RATE) + seg
        elif shift_ms < 0:
            seg = seg[-shift_ms:]

        seg = seg[:target_ms]
        if len(seg) < target_ms:
            seg = seg + AudioSegment.silent(duration=target_ms - len(seg),
                                            frame_rate=SAMPLE_RATE)
        seg = seg.fade_out(40)

        samp = np.array(seg.get_array_of_samples(), dtype=np.float64) / 32768.0
        final_stress = _find_stress_position(samp, SAMPLE_RATE) / SAMPLE_RATE * 1000
        print(f"    '{word}': shift {shift_ms:+d}ms -> stress at {final_stress:.0f}ms")

        normalized[d] = samp

    return normalized


def _strip_silence(audio, silence_thresh=-40, chunk_size=10):
    """Strip leading and trailing silence from an AudioSegment."""
    start = 0
    for i in range(0, len(audio), chunk_size):
        if audio[i:i + chunk_size].dBFS > silence_thresh:
            start = max(0, i - chunk_size)
            break
    end = len(audio)
    for i in range(len(audio) - chunk_size, 0, -chunk_size):
        if audio[i:i + chunk_size].dBFS > silence_thresh:
            end = min(len(audio), i + 2 * chunk_size)
            break
    return audio[start:end]


def main():
    print("=== Pi Day Psy-Trance Generator ===")
    print(f"BPM: {BPM}, Bars: {BARS}")
    print(f"Pi: {PI_INTEGER}.{PI_DECIMALS[:(NUM_PHRASES-1)*DIGITS_PER_GROUP]}")
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

    # === HIHAT — evolving pattern ===
    # Start with simple off-beat 8ths, gradually add 16th note patterns
    print("Generating evolving hihat pattern...")
    for bar in range(BARS):
        bar_start_sample = int(bar * bar_duration_sec * SAMPLE_RATE)
        progress = bar / max(BARS - 1, 1)  # 0→1 over the track

        for sixteenth in range(16):  # 16 sixteenth notes per bar
            pos = bar_start_sample + int(sixteenth * beat_duration_sec / 4 * SAMPLE_RATE)

            # Always play off-beat 8th notes (sixteenth 2, 6, 10, 14)
            is_offbeat_8th = sixteenth % 4 == 2

            # Gradually introduce extra 16th hits
            # After bar 16: add some 16ths. After bar 32: more. After bar 48: full.
            is_extra_16th = False
            if progress > 0.25 and sixteenth in [1, 5, 9, 13]:  # "e" of each beat
                is_extra_16th = np.random.random() < (progress - 0.25) * 0.6
            if progress > 0.5 and sixteenth in [3, 7, 11, 15]:  # "a" of each beat
                is_extra_16th = np.random.random() < (progress - 0.5) * 0.5

            if is_offbeat_8th or is_extra_16th:
                vol = 1.0 if is_offbeat_8th else 0.5 + progress * 0.3
                end = min(pos + len(hihat), total_samples)
                if end > pos:
                    mix[pos:end] += hihat[:end - pos] * vol

    # === BASSLINE per bar (with evolving filter) ===
    print("Generating bassline with filter sweep...")
    for bar in range(BARS):
        bar_start = int(bar * bar_duration_sec * SAMPLE_RATE)
        bar_end = min(bar_start + bar_samples, total_samples)
        actual_bar_samples = bar_end - bar_start
        bass = generate_bassline(bar, actual_bar_samples)
        mix[bar_start:bar_end] += bass[:actual_bar_samples]

    # === PAD — evolving atmospheric layer ===
    print("Generating evolving pad...")
    # Pad plays during rest periods, tonality matches phrase
    # Even phrases: E minor chord tones (E, G, B)
    # Odd phrases: A minor chord tones (A, C, E)
    pad_notes_even = [82.41, 98.00, 123.47]   # E2, G2, B2 (Em)
    pad_notes_odd = [110.00, 130.81, 164.81]   # A2, C3, E3 (Am)
    for phrase in range(NUM_PHRASES):
        progress = phrase / max(NUM_PHRASES - 1, 1)
        if progress < 0.15:
            continue  # No pad in the intro

        phrase_start_sec = phrase * PHRASE_BEATS * beat_duration_sec
        pad_start_sec = phrase_start_sec + SPEAK_BEATS * beat_duration_sec - beat_duration_sec
        pad_duration = (REST_BEATS + 1) * beat_duration_sec
        pad_start_sample = int(pad_start_sec * SAMPLE_RATE)

        pad_notes = pad_notes_even if phrase % 2 == 0 else pad_notes_odd
        note_freq = pad_notes[phrase % len(pad_notes)]
        pad = generate_pad_note(pad_duration, note_freq)
        pad_vol = 0.3 + progress * 0.7

        end = min(pad_start_sample + len(pad), total_samples)
        actual_len = end - pad_start_sample
        if actual_len > 0 and pad_start_sample >= 0:
            mix[pad_start_sample:end] += pad[:actual_len] * pad_vol

    # === RISERS before each phrase (building tension) ===
    print("Generating risers...")
    for phrase in range(1, NUM_PHRASES):  # skip first phrase
        progress = phrase / max(NUM_PHRASES - 1, 1)
        phrase_start_sec = phrase * PHRASE_BEATS * beat_duration_sec

        # Riser during the last part of the rest period (2-3 beats before next phrase)
        riser_duration = beat_duration_sec * 3
        riser_start_sec = phrase_start_sec - riser_duration
        riser_start_sample = int(riser_start_sec * SAMPLE_RATE)

        if riser_start_sample < 0:
            continue

        riser = generate_riser(riser_duration)
        # Risers get more intense as the track progresses
        riser_vol = 0.5 + progress * 0.5

        end = min(riser_start_sample + len(riser), total_samples)
        actual_len = end - riser_start_sample
        if actual_len > 0:
            mix[riser_start_sample:end] += riser[:actual_len] * riser_vol

    # === TTS Pi digits ===
    # Phrase 0: just "3" (integer part), centered in the phrase
    # Phrases 1-15: groups of 5 decimal digits
    print("Generating Pi digit speech (stress-aligned, phrased)...")
    tts_digits = generate_tts_digits()
    digit_interval_sec = beat_duration_sec * BEATS_PER_DIGIT

    # Phrase 0: "three" alone, placed at beat 4 (middle of speak period)
    speech = tts_digits['3'].copy()
    pos = int(4 * beat_duration_sec * SAMPLE_RATE)
    end = min(pos + len(speech), total_samples)
    if end > pos:
        mix[pos:end] += speech[:end - pos] * 0.9
    print(f"    Phrase 1: '3' (integer part)")

    # Phrases 1..15: groups of 5 decimal digits
    decimal_index = 0
    for phrase in range(1, NUM_PHRASES):
        phrase_start_sec = phrase * PHRASE_BEATS * beat_duration_sec

        group = PI_DECIMALS[decimal_index:decimal_index + DIGITS_PER_GROUP]
        print(f"    Phrase {phrase + 1}: '{group}'")

        for d in range(DIGITS_PER_GROUP):
            if decimal_index >= len(PI_DECIMALS):
                break
            digit = PI_DECIMALS[decimal_index]
            speech = tts_digits[digit].copy()

            digit_time = phrase_start_sec + d * digit_interval_sec
            pos = int(digit_time * SAMPLE_RATE)

            # Exhale envelope over 5 digits
            progress = d / max(DIGITS_PER_GROUP - 1, 1)
            if progress < 0.2:
                vol = 0.75 + progress * 1.25
            else:
                vol = 1.0 - progress * 0.25
            vol = max(vol, 0.5)

            end = min(pos + len(speech), total_samples)
            actual_len = end - pos
            if actual_len > 0:
                mix[pos:pos + actual_len] += speech[:actual_len] * 0.9 * vol

            decimal_index += 1

    print(f"    Total: 1 + {decimal_index} = {1 + decimal_index} digits")

    # === Normalize ===
    print("Mixing and normalizing...")
    peak = np.max(np.abs(mix))
    if peak > 0:
        mix = mix / peak * 0.95

    # === Export ===
    mix_16bit = (mix * 32767).astype(np.int16)

    wav_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pi_trance.wav")
    wavfile.write(wav_path, SAMPLE_RATE, mix_16bit)

    mp3_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pi_trance.mp3")
    audio = AudioSegment.from_wav(wav_path)
    audio.export(mp3_path, format="mp3", bitrate="192k",
                 tags={"title": "Pi Trance", "artist": "Pi Day Generator",
                       "album": "3.14159..."})

    os.remove(wav_path)

    print()
    print(f"Done! Output: {mp3_path}")
    print(f"Duration: {total_duration_sec:.1f}s | BPM: {BPM}")
    print(f"Digits: 3.{PI_DECIMALS[:decimal_index]}")


if __name__ == "__main__":
    main()
