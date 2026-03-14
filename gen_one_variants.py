#!/usr/bin/env python3
"""Generate 10 variants of the track with different 'one' stress offsets."""
import subprocess
import shutil
import os

VARIANTS = [
    ("one_v01_minus80", -80),
    ("one_v02_minus65", -65),
    ("one_v03_minus50", -50),
    ("one_v04_minus40", -40),
    ("one_v05_minus30", -30),
    ("one_v06_minus20", -20),
    ("one_v07_minus10", -10),
    ("one_v08_zero",      0),
    ("one_v09_plus10",   10),
    ("one_v10_plus20",   20),
]

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pi_trance.py")
OUTDIR = os.path.dirname(os.path.abspath(__file__))

for name, offset in VARIANTS:
    print(f"\n{'='*50}")
    print(f"  Variant: {name} (one offset={offset:+d}ms)")
    print(f"{'='*50}")

    with open(SCRIPT, 'r') as f:
        code = f.read()

    # Add '1' override while keeping '9': -60
    code = code.replace(
        "stress_overrides = {'9': -60}",
        f"stress_overrides = {{'9': -60, '1': {offset}}}"
    )

    tmp_script = os.path.join(OUTDIR, "_tmp_variant.py")
    with open(tmp_script, 'w') as f:
        f.write(code)

    subprocess.run(["python3", tmp_script], check=True)

    src = os.path.join(OUTDIR, "pi_trance.mp3")
    dst = os.path.join(OUTDIR, f"{name}.mp3")
    shutil.move(src, dst)
    print(f"  -> Saved: {dst}")

os.remove(tmp_script)
print("\nDone! All 10 variants generated.")
