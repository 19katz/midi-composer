"""Encode -> decode -> re-encode every MIDI file and report codec fidelity.

Usage: python scripts/roundtrip.py [MIDI_DIR] [--write-text OUT_DIR]
"""

import argparse
import glob
import io
import os
import warnings

import numpy as np
import pretty_midi

from midicomposer.codec import SPB, encode, decode

warnings.filterwarnings("ignore")


def note_set(pm, spb=SPB):
    step = pm.resolution / spb
    return {
        ("lh" if "left" in inst.name.lower() else "rh", round(pm.time_to_tick(n.start) / step), n.pitch): n.start
        for inst in pm.instruments
        if not inst.is_drum
        for n in inst.notes
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("midi_dir", nargs="?", default=os.path.expanduser("~/MIDI"))
    ap.add_argument("--write-text", help="directory to save encoded .txt files")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.midi_dir, "*.mid")))
    rows = []
    for f in files:
        pm = pretty_midi.PrettyMIDI(f)
        text = encode(pm)
        buf = io.BytesIO()
        decode(text).save(file=buf)
        buf.seek(0)
        pm2 = pretty_midi.PrettyMIDI(buf)
        text2 = encode(pm2)

        a, b = note_set(pm), note_set(pm2)
        matched = a.keys() & b.keys()
        drift = [abs(a[k] - b[k]) for k in matched]
        rows.append(dict(
            f=os.path.basename(f),
            stable=text == text2,
            recall=len(matched) / len(a),
            precision=len(matched) / max(1, len(b)),
            drift_med=float(np.median(drift)) if drift else float("nan"),
            drift_p95=float(np.percentile(drift, 95)) if drift else float("nan"),
            chars=len(text),
        ))
        if args.write_text:
            os.makedirs(args.write_text, exist_ok=True)
            with open(os.path.join(args.write_text, os.path.basename(f)[:-4] + ".txt"), "w") as fh:
                fh.write(text)

    col = lambda k: np.array([r[k] for r in rows])
    print(f"files: {len(rows)}")
    print(f"text stable after round trip: {col('stable').mean():.1%}")
    for k in ("recall", "precision"):
        v = col(k)
        print(f"note {k}: mean={v.mean():.4f} min={v.min():.4f}")
    print(f"onset drift (s): median-of-medians={np.nanmedian(col('drift_med')):.4f} median p95={np.nanmedian(col('drift_p95')):.3f}")
    c = col("chars")
    print(f"chars/piece: median={np.median(c):.0f} p90={np.percentile(c, 90):.0f} max={c.max()} total={c.sum()}")
    worst = sorted(rows, key=lambda r: r["recall"])[:5]
    print("lowest recall:", [(r["f"], round(r["recall"], 4)) for r in worst])
    unstable = [r["f"] for r in rows if not r["stable"]]
    print("unstable:", len(unstable), unstable[:10])


if __name__ == "__main__":
    main()
