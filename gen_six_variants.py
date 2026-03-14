#!/usr/bin/env python3
"""Generate 10 variants of the track with different 'six' stress offsets."""
import subprocess
import shutil
import os

VARIANTS = [
    ("six_v01_minus80", -80),
    ("six_v02_minus65", -65),
    ("six_v03_minus50", -50),
    ("six_v04_minus40", -40),
    ("six_v05_minus30", -30),
    ("six_v06_minus20", -20),
    ("six_v07_minus10", -10),
    ("six_v08_zero",      0),
    ("six_v09_plus10",   10),
    ("six_v10_plus20",   20),
]

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pi_trance.py")
OUTDIR = os.path.dirname(os.path.abspath(__file__))

for name, offset in VARIANTS:
    print(f"\n{'='*50}")
    print(f"  Variant: {name} (six offset={offset:+d}ms)")
    print(f"{'='*50}")

    with open(SCRIPT, 'r') as f:
        code = f.read()

    code = code.replace(
        "stress_overrides = {'9': -60, '1': -40}",
        f"stress_overrides = {{'9': -60, '1': -40, '6': {offset}}}"
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
