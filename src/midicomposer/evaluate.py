"""Cheap automatic checks for generated scores. Listening is still the real test."""

import io
import re

import numpy as np
import pretty_midi

from .codec import HANDS, _key_name, decode
from .dataset import meta_fields, split_piece
from .vocab import parse_pit

_MUSIC = re.compile(r"<\|(?:v|t|d|pit|ped_on|ped_off)_")


def to_pretty_midi(text):
    buf = io.BytesIO()
    decode(text).save(file=buf)
    buf.seek(0)
    return pretty_midi.PrettyMIDI(buf)


def pitch_class_hist(pm):
    h = np.zeros(12)
    for inst in pm.instruments:
        if inst.name == "__dummy_sustain__":
            continue
        for n in inst.notes:
            h[n.pitch % 12] += n.end - n.start
    return h / h.sum() if h.sum() else h


def hand_lines(text):
    return [l for l in text.splitlines() if l.split(" ", 1)[0] in HANDS and len(l.split()) > 1]


def _empty_bar_frac(bars):
    if not bars:
        return 1.0
    empty = 0
    for bar in bars:
        pits = sum(1 for l in bar.splitlines() if l.startswith(HANDS) for t in l.split() if parse_pit(t) is not None)
        empty += pits == 0
    return empty / len(bars)


def evaluate(text, requested_key=None, reference_text=None, train_lines=None):
    meta, bars = split_piece(text)
    toks = [t for l in hand_lines(text) for t in l.split()[1:]]
    valid = sum(bool(_MUSIC.match(t) or t in ("<|ped_on|>", "<|ped_off|>") or parse_pit(t) is not None) for t in toks)
    pm = to_pretty_midi(text)
    n_notes = sum(len(i.notes) for i in pm.instruments if i.name != "__dummy_sustain__")
    out = {
        "bars": len(bars),
        "planned_bars": int(meta_fields(meta).get("bars", 0) or 0),
        "notes": n_notes,
        "notes_per_bar": round(n_notes / max(1, len(bars)), 2),
        "empty_bar_frac": round(_empty_bar_frac(bars), 3),
        "valid_token_frac": valid / max(1, len(toks)),
        "duration_s": round(float(pm.get_end_time()), 1),
    }
    if requested_key:
        want = requested_key.replace(" ", "_").lower()
        out["key_header_match"] = meta_fields(meta).get("key", "").lower() == want
        try:
            out["key_sounding_match"] = _key_name(pm).lower() == want
        except Exception:
            out["key_sounding_match"] = False
        out["key_match"] = out["key_header_match"]  # kept for older result rows
    if reference_text:
        a, b = pitch_class_hist(pm), pitch_class_hist(to_pretty_midi(reference_text))
        out["pc_hist_l1_to_reference"] = float(np.abs(a - b).sum())
    if train_lines is not None:
        lines = [l for l in hand_lines(text) if len(l) > 40]
        out["copied_line_frac"] = sum(l in train_lines for l in lines) / max(1, len(lines))
    return out
