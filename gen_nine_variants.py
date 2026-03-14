#!/usr/bin/env python3
"""Generate 10 variants of the track with different 'nine' stress offsets."""
import subprocess
import shutil
import os

VARIANTS = [
    ("nine_v01_minus60", -60),
    ("nine_v02_minus45", -45),
    ("nine_v03_minus30", -30),
    ("nine_v04_minus20", -20),  # current
    ("nine_v05_minus10", -10),
    ("nine_v06_zero",      0),
    ("nine_v07_plus10",   10),
    ("nine_v08_plus20",   20),
    ("nine_v09_plus30",   30),
    ("nine_v10_plus40",   40),
]

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pi_trance.py")
OUTDIR = os.path.dirname(os.path.abspath(__file__))

for name, offset in VARIANTS:
    print(f"\n{'='*50}")
    print(f"  Variant: {name} (nine offset={offset:+d}ms)")
    print(f"{'='*50}")

    # Read original script
    with open(SCRIPT, 'r') as f:
        code = f.read()

    # Replace the stress override for '9'
    code = code.replace(
        "stress_overrides = {'9': -20}",
        f"stress_overrides = {{'9': {offset}}}"
    )

    # Write temp script
    tmp_script = os.path.join(OUTDIR, "_tmp_variant.py")
    with open(tmp_script, 'w') as f:
        f.write(code)

    subprocess.run(["python3", tmp_script], check=True)

    # Rename output
    src = os.path.join(OUTDIR, "pi_trance.mp3")
    dst = os.path.join(OUTDIR, f"{name}.mp3")
    shutil.move(src, dst)
    print(f"  -> Saved: {dst}")

os.remove(tmp_script)
print("\nDone! All 10 variants generated.")
