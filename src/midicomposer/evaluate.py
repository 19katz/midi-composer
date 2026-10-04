"""Cheap automatic checks for generated scores. Listening is still the real test."""

import io
import re

import numpy as np
import pretty_midi

from .codec import HANDS, _NOTE, decode
from .dataset import meta_fields, split_piece


def to_pretty_midi(text):
    buf = io.BytesIO()
    decode(text).save(file=buf)
    buf.seek(0)
    return pretty_midi.PrettyMIDI(buf)


def pitch_class_hist(pm):
    h = np.zeros(12)
    for inst in pm.instruments:
        for n in inst.notes:
            h[n.pitch % 12] += n.end - n.start
    return h / h.sum() if h.sum() else h


def hand_lines(text):
    return [l for l in text.splitlines() if l.split(" ", 1)[0] in HANDS and len(l) > 2]


def evaluate(text, requested_key=None, reference_text=None, train_lines=None):
    meta, bars = split_piece(text)
    toks = [t for l in hand_lines(text) for t in l.split()[1:]]
    valid = sum(bool(_NOTE.match(t) or re.fullmatch(r"v\d+", t)) for t in toks)
    pm = to_pretty_midi(text)
    out = {
        "bars": len(bars),
        "planned_bars": int(meta_fields(meta).get("bars", 0) or 0),
        "notes": sum(len(i.notes) for i in pm.instruments),
        "valid_token_frac": valid / max(1, len(toks)),
        "duration_s": round(float(pm.get_end_time()), 1),
    }
    if requested_key:
        out["key_match"] = meta_fields(meta).get("key", "").lower() == requested_key.replace(" ", "_").lower()
    if reference_text:
        a, b = pitch_class_hist(pm), pitch_class_hist(to_pretty_midi(reference_text))
        out["pc_hist_l1_to_reference"] = float(np.abs(a - b).sum())
    if train_lines is not None:
        lines = [l for l in hand_lines(text) if len(l) > 40]
        out["copied_line_frac"] = sum(l in train_lines for l in lines) / max(1, len(lines))
    return out
