"""Musical features that ground the description LLM."""

import mido
import numpy as np
import pretty_midi

from .codec import _key_name


def embedded_metadata(path):
    """Title, composer, tempo marking, etc. from track-0 text events (piano-midi.de convention)."""
    mid = mido.MidiFile(path)
    out = []
    for msg in mid.tracks[0]:
        if msg.type == "track_name":
            out.append(msg.name)
        elif msg.type == "text":
            out.append(msg.text)
    skip = ("fertiggestellt", "erstellt", "veröffentlicht", "update", "normierung", "dauer", "copyright", "©")
    return [s.strip() for s in out if s.strip() and not any(w in s.lower() for w in skip)][:6]


def extract_features(path):
    pm = pretty_midi.PrettyMIDI(path)
    notes = [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    downbeats = list(pm.get_downbeats()) + [pm.get_end_time()]
    _, tempi = pm.get_tempo_changes()
    per_bar = []
    for b in range(len(downbeats) - 1):
        bn = [n for n in notes if downbeats[b] <= n.start < downbeats[b + 1]]
        per_bar.append({
            "bar": b + 1,
            "notes": len(bn),
            "vel": round(float(np.mean([n.velocity for n in bn]))) if bn else 0,
            "lo": min(n.pitch for n in bn) if bn else None,
            "hi": max(n.pitch for n in bn) if bn else None,
        })
    key = _key_name(pm).replace("_", " ")
    return {
        "metadata": embedded_metadata(path),
        "key": key,
        "time_signatures": sorted({f"{t.numerator}/{t.denominator}" for t in pm.time_signature_changes}),
        "tempo_bpm_range": [round(float(np.percentile(tempi, 5))), round(float(np.percentile(tempi, 95)))],
        "duration_s": round(pm.get_end_time()),
        "n_bars": len(per_bar),
        "per_bar": per_bar,
    }
