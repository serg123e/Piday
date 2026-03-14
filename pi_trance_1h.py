#!/usr/bin/env python3
"""
Pi Trance — 1 Hour Edition
Generates a 1-hour psy-trance track with spoken digits of Pi.
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
TARGET_DURATION_SEC = 3600  # 1 hour

BEATS_PER_BAR = 4
beat_duration_sec = 60.0 / BPM
bar_duration_sec = beat_duration_sec * BEATS_PER_BAR

BARS = int(TARGET_DURATION_SEC / bar_duration_sec)
# Round to multiple of 4 (phrase length)
BARS = (BARS // 4) * 4
TOTAL_BEATS = BARS * BEATS_PER_BAR

BEATS_PER_DIGIT = 2
DIGITS_PER_GROUP = 5
SPEAK_BEATS = DIGITS_PER_GROUP * BEATS_PER_DIGIT
REST_BEATS = 6
PHRASE_BEATS = SPEAK_BEATS + REST_BEATS
PHRASE_BARS = PHRASE_BEATS // BEATS_PER_BAR
NUM_PHRASES = BARS // PHRASE_BARS
TOTAL_DIGITS = 1 + (NUM_PHRASES - 1) * DIGITS_PER_GROUP

total_duration_sec = bar_duration_sec * BARS
total_samples = int(total_duration_sec * SAMPLE_RATE)

# Pi digits — need ~2600+
PI_INTEGER = "3"
PI_DECIMALS = (
    "14159265358979323846264338327950288419716939937510"
    "58209749445923078164062862089986280348253421170679"
    "82148086513282306647093844609550582231725359408128"
    "48111745028410270193852110555964462294895493038196"
    "44288109756659334461284756482337867831652712019091"
    "45648566923460348610454326648213393607260249141273"
    "72458700660631558817488152092096282925409171536436"
    "78925903600113305305488204665213841469519415116094"
    "33057270365759591953092186117381932611793105118548"
    "07446237996274956735188575272489122793818301194912"
    "98336733624406566430860213949463952247371907021798"
    "60943702770539217176293176752384674818467669405132"
    "00056812714526356082778577134275778960917363717872"
    "14684409012249534301465495853710507922796892589235"
    "42019956112129021960864034418159813629774771309960"
    "51870721134999999837297804995105973173281609631859"
    "50244594553469083026425223082533446850352619311881"
    "71010003137838752886587533208381420617177669147303"
    "59825349042875546873115956286388235378759375195778"
    "18577805321712268066130019278766111959092164201989"
    "38095257201065485863278865936153381827968230301952"
    "03530185296899577362259941389124972177528347913151"
    "55748572424541506959508295331168617278558890750983"
    "81754637464939319255060400927701671139009848824012"
    "85836160356370766010471018194295559619894676783744"
    "94482553797747268471040475346462080466842590694912"
    "93313677028989152104752162056966024058038150193511"
    "25338243003558764024749647326391419927260426992279"
    "67823547816360093417216412199245863150302861829745"
    "55706749838505494588586926995690927210797509302955"
    "32116534498720275596023648066549911988183479775356"
    "63698074265425278625518184175746728909777727938000"
    "81647060016145249192173217214772350141441973568548"
    "16136115735255213347574184946843852332390739414333"
    "45477624168625189835694855620992192221842725502542"
    "56887671790494601653466804988627232791786085784383"
    "82796797668145410095388378636095068006422512520511"
    "73929848960841284886269456042419652850222106611863"
    "06744278622039194945047123713786960956364371917287"
    "46776465757396241389086583264599581339047802759009"
    "94657640789512694683983525957098258226205224894077"
    "26719478268482601476990902640136394437455305068203"
    "49625245174939965143142980919065925093722169646151"
    "57098583874105978859597729754989301617539284681382"
    "68683868942774155991855925245953959431049972524680"
    "84598727364469584865383673622262609912460805124388"
    "43904512441365497627807977156914359977001296160894"
    "41694868555848406353422072225828488648158456028506"
    "01684273945226746767889525213852254995466672782398"
    "64565961163548862305774564980355936345681743241125"
    "15003546345458858796575272060963023701325813568643"
    "42576564560167805916963287539688863883065817489555"
    "75677122809876921112785026843550745953614862062652"
)


def generate_kick(duration_sec=0.15, sample_rate=SAMPLE_RATE):
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    freq_start, freq_end = 160, 38
    freq = freq_start * np.exp(-t * np.log(freq_start / freq_end) / duration_sec)
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate
    envelope = np.exp(-t * 25)
    click = np.exp(-t * 200) * 0.5
    return (np.sin(phase) * envelope + click * np.sin(2 * np.pi * 1000 * t)) * 0.8


def get_phrase_for_bar(bar_index):
    return bar_index // PHRASE_BARS


def generate_bassline(bar_index, bar_samples, sample_rate=SAMPLE_RATE):
    phrase = get_phrase_for_bar(bar_index)
    bar_in_phrase = bar_index % PHRASE_BARS
    if phrase % 2 == 0:
        base_freqs = [41.20, 41.20, 49.00, 41.20, 46.25, 41.20, 49.00, 55.00]
    else:
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
        note = np.sin(2 * np.pi * freq * t_note)
        note += 0.6 * np.sin(2 * np.pi * freq * 2 * t_note)
        note += 0.35 * np.sin(2 * np.pi * freq * 3 * t_note)
        note += 0.2 * np.sin(2 * np.pi * freq * 4 * t_note)
        note += 0.1 * np.sin(2 * np.pi * freq * 5 * t_note)
        note *= np.exp(-t_note * 15)
        bass[start:end] += note[:end - start]

    for beat in range(BEATS_PER_BAR):
        duck_start = int(beat * beat_duration_sec * sample_rate)
        duck_len = int(0.08 * sample_rate)
        duck_end = min(duck_start + duck_len, bar_samples)
        duck_samples = duck_end - duck_start
        if duck_samples > 0:
            bass[duck_start:duck_end] *= np.linspace(0.1, 1.0, duck_samples)

    # Filter sweep cycles every ~64 bars
    cycle_pos = (bar_index % 64) / 63
    cutoff = 300 + cycle_pos * 1500
    nyq = sample_rate / 2
    b, a = butter(2, min(cutoff / nyq, 0.99), btype='low')
    bass = lfilter(b, a, bass)
    return bass * 0.35


def generate_hihat(duration_sec=0.05, sample_rate=SAMPLE_RATE):
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    return np.random.randn(n_samples) * np.exp(-t * 80) * 0.15


def generate_riser(duration_sec, sample_rate=SAMPLE_RATE):
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    noise = np.random.randn(n_samples)
    envelope = (t / duration_sec) ** 2
    riser = np.zeros(n_samples)
    chunk_size = int(0.02 * sample_rate)
    for i in range(0, n_samples - chunk_size, chunk_size):
        progress = i / n_samples
        cutoff = 500 + progress * 8000
        nyq = sample_rate / 2
        b, a = butter(2, min(cutoff / nyq, 0.99), btype='low')
        riser[i:i + chunk_size] = lfilter(b, a, noise[i:i + chunk_size])
    riser *= envelope
    return riser * 0.15


def generate_pad_note(duration_sec, freq, sample_rate=SAMPLE_RATE):
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    pad = np.sin(2 * np.pi * freq * t)
    pad += np.sin(2 * np.pi * freq * 1.003 * t)
    pad += 0.5 * np.sin(2 * np.pi * freq * 2.001 * t)
    attack = int(0.3 * sample_rate)
    release = int(0.5 * sample_rate)
    envelope = np.ones(n_samples)
    if attack < n_samples:
        envelope[:attack] = np.linspace(0, 1, attack)
    if release < n_samples:
        envelope[-release:] = np.linspace(1, 0, release)
    lfo = 1.0 + 0.15 * np.sin(2 * np.pi * 0.25 * t)
    pad *= envelope * lfo
    return pad * 0.06


def _find_stress_position(samples, sample_rate):
    window = int(0.020 * sample_rate)
    hop = window // 4
    rms = np.array([np.sqrt(np.mean(samples[i:i + window] ** 2))
                     for i in range(0, len(samples) - window, hop)])
    return np.argmax(rms) * hop


def _strip_silence(audio, silence_thresh=-40, chunk_size=10):
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


def generate_tts_digits():
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
        stress_ms = _find_stress_position(samp, SAMPLE_RATE) / SAMPLE_RATE * 1000
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
    stress_overrides = {'9': -60, '1': -40, '6': -40, '8': -10}

    normalized = {}
    for d, (seg, stress_ms) in sped_up.items():
        override = stress_overrides.get(d, 0)
        shift_ms = int(STRESS_TARGET_MS + override - stress_ms)
        if shift_ms > 0:
            seg = AudioSegment.silent(duration=shift_ms, frame_rate=SAMPLE_RATE) + seg
        elif shift_ms < 0:
            seg = seg[-shift_ms:]
        seg = seg[:target_ms]
        if len(seg) < target_ms:
            seg = seg + AudioSegment.silent(duration=target_ms - len(seg), frame_rate=SAMPLE_RATE)
        seg = seg.fade_out(40)
        normalized[d] = np.array(seg.get_array_of_samples(), dtype=np.float64) / 32768.0

    return normalized


def main():
    print("=== Pi Trance — 1 Hour Edition ===")
    print(f"BPM: {BPM}, Bars: {BARS}, Phrases: {NUM_PHRASES}")
    print(f"Digits needed: {TOTAL_DIGITS}, available: {len(PI_DECIMALS)}")
    print(f"Duration: {total_duration_sec:.1f}s = {total_duration_sec/60:.1f} min")
    print()

    if TOTAL_DIGITS - 1 > len(PI_DECIMALS):
        print(f"WARNING: need {TOTAL_DIGITS-1} decimal digits but only have {len(PI_DECIMALS)}")
        print("Will loop digits.")

    kick = generate_kick()
    hihat = generate_hihat()

    # Process in chunks to manage memory (~158M samples for 1 hour)
    # We'll build the mix in segments of 64 bars each
    CHUNK_BARS = 64
    num_chunks = (BARS + CHUNK_BARS - 1) // CHUNK_BARS

    bar_samples = int(bar_duration_sec * SAMPLE_RATE)

    print("Generating TTS digits...")
    tts_digits = generate_tts_digits()
    digit_interval_sec = beat_duration_sec * BEATS_PER_DIGIT

    pad_notes_even = [82.41, 98.00, 123.47]
    pad_notes_odd = [110.00, 130.81, 164.81]

    # Output file path
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pi_trance_1h.mp3")
    wav_path = out_path.replace('.mp3', '.wav')

    # Build full mix
    print(f"\nGenerating {BARS} bars in {num_chunks} chunks of {CHUNK_BARS}...")
    mix = np.zeros(total_samples, dtype=np.float32)

    decimal_index = 0

    for chunk_idx in range(num_chunks):
        bar_start_idx = chunk_idx * CHUNK_BARS
        bar_end_idx = min(bar_start_idx + CHUNK_BARS, BARS)
        print(f"  Chunk {chunk_idx+1}/{num_chunks}: bars {bar_start_idx}-{bar_end_idx-1}")

        for bar in range(bar_start_idx, bar_end_idx):
            bar_start_sample = int(bar * bar_duration_sec * SAMPLE_RATE)
            bar_end_sample = min(bar_start_sample + bar_samples, total_samples)
            actual_bar_samples = bar_end_sample - bar_start_sample

            # KICK
            for beat in range(BEATS_PER_BAR):
                pos = bar_start_sample + int(beat * beat_duration_sec * SAMPLE_RATE)
                end = min(pos + len(kick), total_samples)
                if end > pos:
                    mix[pos:end] += kick[:end - pos]

            # HIHAT
            progress = (bar % 64) / 63
            for s in range(16):
                pos = bar_start_sample + int(s * beat_duration_sec / 4 * SAMPLE_RATE)
                is_offbeat = s % 4 == 2
                is_extra = False
                if progress > 0.25 and s in [1, 5, 9, 13]:
                    is_extra = np.random.random() < (progress - 0.25) * 0.6
                if progress > 0.5 and s in [3, 7, 11, 15]:
                    is_extra = np.random.random() < (progress - 0.5) * 0.5
                if is_offbeat or is_extra:
                    vol = 1.0 if is_offbeat else 0.5 + progress * 0.3
                    end = min(pos + len(hihat), total_samples)
                    if end > pos:
                        mix[pos:end] += hihat[:end - pos] * vol

            # BASS
            bass = generate_bassline(bar, actual_bar_samples)
            mix[bar_start_sample:bar_end_sample] += bass[:actual_bar_samples]

        # PAD + RISER + VOX per phrase in this chunk
        for phrase in range(NUM_PHRASES):
            phrase_bar_start = phrase * PHRASE_BARS
            if phrase_bar_start < bar_start_idx or phrase_bar_start >= bar_end_idx:
                continue

            phrase_start_sec = phrase * PHRASE_BEATS * beat_duration_sec
            # Cycle progress for evolving effects
            cycle_progress = (phrase % 16) / 15

            # PAD (skip first 2 phrases)
            if phrase > 1:
                pad_start_sec = phrase_start_sec + SPEAK_BEATS * beat_duration_sec - beat_duration_sec
                pad_duration = (REST_BEATS + 1) * beat_duration_sec
                pad_start_sample = int(pad_start_sec * SAMPLE_RATE)
                pad_notes = pad_notes_even if phrase % 2 == 0 else pad_notes_odd
                note_freq = pad_notes[phrase % len(pad_notes)]
                pad = generate_pad_note(pad_duration, note_freq)
                pad_vol = 0.3 + cycle_progress * 0.7
                end = min(pad_start_sample + len(pad), total_samples)
                actual_len = end - pad_start_sample
                if actual_len > 0 and pad_start_sample >= 0:
                    mix[pad_start_sample:end] += pad[:actual_len] * pad_vol

            # RISER (skip first phrase)
            if phrase > 0:
                riser_duration = beat_duration_sec * 3
                riser_start_sec = phrase_start_sec - riser_duration
                riser_start_sample = int(riser_start_sec * SAMPLE_RATE)
                if riser_start_sample >= 0:
                    riser = generate_riser(riser_duration)
                    riser_vol = 0.5 + cycle_progress * 0.5
                    end = min(riser_start_sample + len(riser), total_samples)
                    actual_len = end - riser_start_sample
                    if actual_len > 0:
                        mix[riser_start_sample:end] += riser[:actual_len] * riser_vol

    # VOX — separate pass for correct digit ordering
    print("\nPlacing digit speech...")
    decimal_index = 0

    # Phrase 0: "three" at beat 4
    pos = int(4 * beat_duration_sec * SAMPLE_RATE)
    speech = tts_digits['3'].copy()
    end = min(pos + len(speech), total_samples)
    if end > pos:
        mix[pos:end] += speech[:end - pos] * 0.9
    print(f"  Phrase 1: '3'")

    for phrase in range(1, NUM_PHRASES):
        phrase_start_sec = phrase * PHRASE_BEATS * beat_duration_sec
        group_start = decimal_index
        group = ""

        for d in range(DIGITS_PER_GROUP):
            if decimal_index >= len(PI_DECIMALS):
                decimal_index = 0  # loop
            digit = PI_DECIMALS[decimal_index]
            speech = tts_digits[digit].copy()

            digit_time = phrase_start_sec + d * digit_interval_sec
            pos = int(digit_time * SAMPLE_RATE)

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

            group += digit
            decimal_index += 1

        if phrase <= 5 or phrase % 50 == 0:
            print(f"  Phrase {phrase+1}: '{group}'")

    print(f"  Total decimal digits used: {decimal_index}")

    # Normalize
    print("\nNormalizing...")
    peak = np.max(np.abs(mix))
    if peak > 0:
        mix = mix / peak * 0.95

    # Export
    print("Writing WAV...")
    mix_16bit = (mix * 32767).astype(np.int16)
    wavfile.write(wav_path, SAMPLE_RATE, mix_16bit)

    print("Converting to MP3 (192k)...")
    audio = AudioSegment.from_wav(wav_path)
    audio.export(out_path, format="mp3", bitrate="192k",
                 tags={"title": "Pi Trance (1 Hour)", "artist": "Pi Day Generator",
                       "album": "3.14159...", "year": "2025"})
    os.remove(wav_path)

    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\nDone! Output: {out_path}")
    print(f"Duration: {total_duration_sec/60:.1f} min | Size: {file_size_mb:.1f} MB")
    print(f"Digits: 3.{PI_DECIMALS[:min(decimal_index, 50)]}...")


if __name__ == "__main__":
    main()
