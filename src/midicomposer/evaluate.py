"""Cheap automatic checks for generated scores. Listening is still the real test."""

import io
import re

import numpy as np
import pretty_midi

from .codec import HANDS, _key_name, _ts_steps, decode, parse_hand_notes
from .dataset import meta_fields, split_piece
from .vocab import parse_pit, parse_tagged_int

_NOTE_LINES = ("<|mel|>",) + HANDS
_MUSIC = re.compile(r"<\|(?:v|t|d|pit|sd|oct|acc|ped_on|ped_off)_")


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
    return [l for l in text.splitlines() if l.split(" ", 1)[0] in _NOTE_LINES and len(l.split()) > 1]


_MAJOR = {0, 2, 4, 5, 7, 9, 11}
_MINOR = {0, 2, 3, 5, 7, 8, 10}  # natural minor


def _scale(key):
    number = pretty_midi.key_name_to_key_number(key.replace("_", " "))
    return (number % 12, _MINOR if number >= 12 else _MAJOR)


def _bar_events(bar, bar_len, key=None):
    """(onset, duration, pitch, hand). Melody counts as the right hand.

    Onsets are not clamped here, so a note that starts past the bar still shows
    up as silence. Decode is what clamps those onsets for playback.
    """
    events = []
    for line in bar.splitlines()[1:]:
        tag = line.split(" ", 1)[0]
        if tag not in _NOTE_LINES:
            continue
        hand = "<|rh|>" if tag == "<|mel|>" else tag
        for onset, dur, pitch, _vel in parse_hand_notes(line.split()[1:], key):
            events.append((onset, dur, pitch, hand))
    return events


def surface_metrics(text, key=None):
    """Key, rhythm, silence, and tempo measures, from the score text alone.

    diatonic_frac: share of note-duration whose pitch class is in the key's scale
    (major, or natural minor). Duration is clipped to the bar length.
    rhythm_repeat_frac: share of right-hand bars whose (onset, duration) list
    equals the previous bar.
    silence_frac: mean fraction of each bar's grid with no sounding note.
    tempo_jump_frac: share of bar-to-bar tempo changes larger than 8%.
    """
    meta, bars = split_piece(text)
    fields = meta_fields(meta)
    key = key or fields.get("key")
    scale = None
    if key:
        try:
            scale = _scale(key)
        except Exception:
            scale = None
    ts = fields.get("ts", "4/4")
    dia = total = 0.0
    silence, tempos = [], []
    same = pairs = 0
    prev_rh = None
    for bar in bars:
        head = bar.splitlines()[0]
        bar_fields = dict(re.findall(r"(\w+)=(\S+)", head))
        if "ts" in bar_fields:
            ts = bar_fields["ts"]
        try:
            bar_len = int(bar_fields["len"]) if "len" in bar_fields else _ts_steps(ts, int(fields.get("spb", 12)))
        except ValueError:
            bar_len = _ts_steps("4/4", 12)
        bar_len = max(1, bar_len)
        q = next((parse_tagged_int(t, "q") for t in head.split() if t.startswith("<|q_")), None)
        # Hold repeats the previous tempo, so a held bar counts as "no jump"
        # rather than disappearing from the denominator.
        if q:
            bpm = q
            tempos.append(bpm)
        elif "<|tempo_hold|>" in head and tempos:
            tempos.append(tempos[-1])
        events = _bar_events(bar, bar_len, key)
        covered = [False] * bar_len
        for onset, dur, pitch, _hand in events:
            weight = max(0, min(onset + dur, bar_len) - max(0, onset))
            total += weight
            if scale and (pitch - scale[0]) % 12 in scale[1]:
                dia += weight
            for step in range(max(0, onset), min(bar_len, onset + dur)):
                covered[step] = True
        silence.append(1 - sum(covered) / bar_len)
        rh = tuple((onset, dur) for onset, dur, _pitch, hand in events if hand == "<|rh|>")
        if prev_rh is not None and rh:
            pairs += 1
            same += rh == prev_rh
        prev_rh = rh
    jumps = [abs(b - a) / a > 0.08 for a, b in zip(tempos, tempos[1:]) if a]
    return {
        "diatonic_frac": round(dia / total, 3) if scale and total else None,
        "rhythm_repeat_frac": round(same / pairs, 3) if pairs else None,
        "silence_frac": round(sum(silence) / len(silence), 3) if silence else None,
        "tempo_jump_frac": round(sum(jumps) / len(jumps), 3) if jumps else None,
    }


def _empty_bar_frac(bars):
    if not bars:
        return 1.0
    empty = 0
    for bar in bars:
        pits = sum(len(parse_hand_notes(l.split()[1:], None)) for l in bar.splitlines() if l.split(" ", 1)[0] in _NOTE_LINES)
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
        **surface_metrics(text, requested_key or meta_fields(meta).get("key")),
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
