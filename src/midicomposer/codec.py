"""MIDI <-> text codec for score-aligned solo piano MIDI.

Format (each musical field is one vocabulary token; see vocab.py):

    <|piece_start|>
    <|meta|> key=F_minor ts=12/8 bars=262 spb=12
    <|plan|> Exposition:1-65 Development:66-135 ...      (optional; ignored by decode)
    <|bar|> 1 <|pos_0|> <|q_126|> ts=12/8
    <|rh|> <|v_6|> <|t_0|> <|pit_C5|> <|d_6|> <|v_5|> <|t_6|> <|pit_Ab4|> <|pit_C5|> <|d_6|>
    <|lh|> <|v_4|> <|t_0|> <|pit_F2|> <|pit_C3|> <|d_36|>
    <|ped|> <|t_0|> <|ped_on|> <|t_18|> <|ped_off|>
    ...
    <|piece_end|>

Bar lines: bar number (ordinary digits), `<|pos_N|>` = N% through the piece, `<|q_N|>` =
average quarter-note BPM over the bar. `ts=` appears only when the time signature
changes, and `len=` only when the bar's length in grid steps differs from what its
time signature implies. Rubato within a bar is dropped.

Hand lines: `<|v_N|>` sets the velocity bin (0-7) for the events that follow.
Each event is `<|t_onset|>` then one or more `<|pit_...|>` then `<|d_duration|>`.
Onset is relative to the bar; onset and duration are in grid steps (`spb` steps per
quarter note). Sustain pedal (CC64, binarized) is an optional `<|ped|>` line.
"""

import bisect
import re
from collections import defaultdict

import mido
import numpy as np
import pretty_midi

from .vocab import (
    d_on,
    parse_pit,
    parse_tagged_int,
    pit,
    pos,
    t_on,
    tempo,
    vel,
)

SPB = 12
VEL_BINS = 8
TICKS_PER_STEP = 40
PIANO_RANGE = (21, 108)
PEDAL_CC = 64

_MIDO_KEYS = [
    "C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B",
    "Cm", "C#m", "Dm", "Ebm", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "Bbm", "Bm",
]

HANDS = ("<|rh|>", "<|lh|>")
TRACK_NAMES = {"<|rh|>": "Piano right", "<|lh|>": "Piano left"}


def _hand(inst):
    name = inst.name.lower()
    if name == "__dummy_sustain__":
        return None
    if "left" in name or "ped" in name:
        return "<|lh|>"
    return "<|rh|>"


_KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def _key_name(pm):
    """Main key from the first key signature. These files never mark minor mode, so choose
    between the signature's major key and its relative minor: the final bar's bass note
    decides when it is one of the two tonics, otherwise Krumhansl-Schmuckler correlation."""
    if not pm.key_signature_changes:
        return "unknown"
    marked = pm.key_signature_changes[0].key_number
    if marked >= 12:
        return pretty_midi.key_number_to_key_name(marked).replace(" ", "_")
    notes = [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    if not notes:
        return pretty_midi.key_number_to_key_name(marked).replace(" ", "_")
    major = marked
    minor = (major + 9) % 12
    last_downbeat = pm.get_downbeats()[-1] if len(pm.get_downbeats()) else 0
    final = [n for n in notes if n.start >= last_downbeat] or notes[-8:]
    bass = min(final, key=lambda n: n.pitch).pitch % 12
    if bass in (major, minor) and major != minor:
        is_minor = bass == minor
    else:
        hist = np.zeros(12)
        for n in notes:
            hist[n.pitch % 12] += n.end - n.start
        corr = lambda profile, tonic: np.corrcoef(hist, np.roll(profile, tonic))[0, 1]
        is_minor = corr(_KS_MINOR, minor) > corr(_KS_MAJOR, major)
    number = minor + 12 if is_minor else major
    return pretty_midi.key_number_to_key_name(number).replace(" ", "_")


def _ts_steps(ts, spb):
    num, den = map(int, ts.split("/"))
    return round(num * 4 / den * spb)


def _pedal_steps(pm, step):
    evs = []
    for inst in pm.instruments:
        for c in inst.control_changes:
            if c.number == PEDAL_CC:
                evs.append((round(pm.time_to_tick(c.time) / step), 1 if c.value >= 64 else 0))
    evs.sort()
    out, last = [], None
    for t, on in evs:
        if on != last:
            out.append((t, on))
            last = on
    return out


def encode(pm, spb=SPB, vel_bins=VEL_BINS):
    tpb = pm.resolution
    step = tpb / spb
    notes = [(inst, n) for inst in pm.instruments if not inst.is_drum and _hand(inst) for n in inst.notes]
    end_tick = max((pm.time_to_tick(n.end) for _, n in notes), default=1)

    bar_ticks = sorted({pm.time_to_tick(t) for t in pm.get_downbeats()})
    bar_ticks = [t for t in bar_ticks if t < end_tick] or [0]
    n_bars = len(bar_ticks)
    bar_ticks.append(max(end_tick, bar_ticks[-1] + 1))
    bar_steps = [round(t / step) for t in bar_ticks]

    ts_changes = sorted(
        (pm.time_to_tick(ts.time), f"{ts.numerator}/{ts.denominator}") for ts in pm.time_signature_changes
    ) or [(0, "4/4")]
    ts_ticks = [t for t, _ in ts_changes]

    events = {h: [defaultdict(list) for _ in range(n_bars)] for h in HANDS}
    for inst, n in notes:
        s = round(pm.time_to_tick(n.start) / step)
        dur = max(1, round(pm.time_to_tick(n.end) / step) - s)
        b = min(max(bisect.bisect_right(bar_steps, s) - 1, 0), n_bars - 1)
        vel_bin = min(vel_bins - 1, n.velocity * vel_bins // 128)
        events[_hand(inst)][b][(s - bar_steps[b], dur, vel_bin)].append(n.pitch)

    ped_by_bar = [[] for _ in range(n_bars)]
    for s, on in _pedal_steps(pm, step):
        b = min(max(bisect.bisect_right(bar_steps, s) - 1, 0), n_bars - 1)
        ped_by_bar[b].append((s - bar_steps[b], on))

    lines = ["<|piece_start|>", f"<|meta|> key={_key_name(pm)} ts={ts_changes[0][1]} bars={n_bars} spb={spb}"]
    prev_ts = None
    for b in range(n_bars):
        t0, t1 = bar_ticks[b], bar_ticks[b + 1]
        ts = ts_changes[max(bisect.bisect_right(ts_ticks, t0) - 1, 0)][1]
        seconds = pm.tick_to_time(t1) - pm.tick_to_time(t0)
        bpm = round(60 * (t1 - t0) / tpb / seconds) if seconds > 0 else 120
        fields = ["<|bar|>", str(b + 1), pos(b / n_bars), tempo(bpm)]
        if ts != prev_ts:
            fields.append(f"ts={ts}")
        prev_ts = ts
        length = bar_steps[b + 1] - bar_steps[b]
        if length != _ts_steps(ts, spb) and b < n_bars - 1:
            fields.append(f"len={length}")
        lines.append(" ".join(fields))
        for h in HANDS:
            toks, cur_vel = [h], None
            for (on, dur, vel_bin), pitches in sorted(events[h][b].items()):
                if vel_bin != cur_vel:
                    toks.append(vel(vel_bin))
                    cur_vel = vel_bin
                toks.append(t_on(on))
                toks.extend(pit(p) for p in sorted(set(pitches)))
                toks.append(d_on(dur))
            lines.append(" ".join(toks))
        if ped_by_bar[b]:
            ped = ["<|ped|>"]
            for on, state in ped_by_bar[b]:
                ped += [t_on(on), "<|ped_on|>" if state else "<|ped_off|>"]
            lines.append(" ".join(ped))
    lines.append("<|piece_end|>")
    return "\n".join(lines)


_FIELD = re.compile(r"(\w+)=(\S+)")


def decode(text, vel_bins=VEL_BINS):
    """Parse codec text into a mido.MidiFile. Malformed tokens are skipped."""
    spb, ts, bpm, key = SPB, "4/4", 120, None
    bar_start, bar_len, in_bar = 0, None, False
    tempo_events, ts_events, ped_events = [], [], []
    notes = {h: [] for h in HANDS}

    for line in text.splitlines():
        toks = line.strip().split()
        if not toks:
            continue
        tag, rest = toks[0], toks[1:]
        fields = dict(_FIELD.findall(line))
        if tag == "<|meta|>":
            spb = int(fields.get("spb", spb))
            ts = fields.get("ts", ts)
            key = fields.get("key")
        elif tag == "<|bar|>":
            if in_bar:
                bar_start += bar_len
            in_bar = True
            if "ts" in fields and fields["ts"] != (ts_events[-1][1] if ts_events else None):
                ts = fields["ts"]
                ts_events.append((bar_start, ts))
            elif not ts_events:
                ts_events.append((bar_start, ts))
            try:
                bar_len = int(fields["len"]) if "len" in fields else _ts_steps(ts, spb)
            except ValueError:
                bar_len = _ts_steps("4/4", spb)
            q = next((parse_tagged_int(t, "q") for t in rest if t.startswith("<|q_")), None)
            if q:
                bpm = q
            tempo_events.append((bar_start, bpm))
        elif tag in HANDS and in_bar:
            vel_bin, onset, pitches = vel_bins // 2, 0, []
            for tok in rest:
                if tok.startswith("<|v_"):
                    vel_bin = parse_tagged_int(tok, "v") or vel_bin
                elif tok.startswith("<|t_"):
                    onset = parse_tagged_int(tok, "t") or 0
                    pitches = []
                elif tok.startswith("<|pit_"):
                    p = parse_pit(tok)
                    if p is not None:
                        pitches.append(p)
                elif tok.startswith("<|d_"):
                    dur = max(1, parse_tagged_int(tok, "d") or 1)
                    velocity = min(127, vel_bin * (128 // vel_bins) + 128 // vel_bins // 2)
                    for p in pitches:
                        if 0 <= p <= 127:
                            notes[tag].append((bar_start + onset, dur, p, velocity))
                    pitches = []
        elif tag == "<|ped|>" and in_bar:
            onset = 0
            for tok in rest:
                if tok.startswith("<|t_"):
                    onset = parse_tagged_int(tok, "t") or 0
                elif tok == "<|ped_on|>":
                    ped_events.append((bar_start + onset, 127))
                elif tok == "<|ped_off|>":
                    ped_events.append((bar_start + onset, 0))

    tpb = spb * TICKS_PER_STEP
    mid = mido.MidiFile(ticks_per_beat=tpb)
    meta = [(s * TICKS_PER_STEP, 0, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(t))) for s, t in tempo_events]
    for s, sig in ts_events:
        num, den = map(int, sig.split("/"))
        meta.append((s * TICKS_PER_STEP, 0, mido.MetaMessage("time_signature", numerator=num, denominator=den)))
    try:
        key_number = pretty_midi.key_name_to_key_number(key.replace("_", " "))
        meta.append((0, 0, mido.MetaMessage("key_signature", key=_MIDO_KEYS[key_number])))
    except Exception:
        pass
    mid.tracks.append(_to_track(meta))
    for h in HANDS:
        evs = []
        for on, dur, pitch, velocity in notes[h]:
            evs.append((on * TICKS_PER_STEP, 1, mido.Message("note_on", note=pitch, velocity=velocity)))
            evs.append(((on + dur) * TICKS_PER_STEP, 0, mido.Message("note_off", note=pitch, velocity=0)))
        track = _to_track(evs)
        track.insert(0, mido.MetaMessage("track_name", name=TRACK_NAMES[h]))
        mid.tracks.append(track)
    if ped_events:
        # pretty_midi drops CC-only tracks; a dummy note keeps the pedal track readable.
        evs = [
            (0, 1, mido.Message("note_on", note=21, velocity=1)),
            (TICKS_PER_STEP, 0, mido.Message("note_off", note=21, velocity=0)),
        ]
        evs.extend(
            (s * TICKS_PER_STEP, 0, mido.Message("control_change", control=PEDAL_CC, value=val))
            for s, val in ped_events
        )
        track = _to_track(evs)
        track.insert(0, mido.MetaMessage("track_name", name="__dummy_sustain__"))
        mid.tracks.append(track)
    return mid


def _to_track(events):
    track, now = mido.MidiTrack(), 0
    for tick, _, msg in sorted(events, key=lambda e: (e[0], e[1])):
        track.append(msg.copy(time=tick - now))
        now = tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    return track


def transpose_key(key, semis):
    try:
        number = pretty_midi.key_name_to_key_number(key.replace("_", " "))
    except Exception:
        return key
    mode = number // 12
    return pretty_midi.key_number_to_key_name((number % 12 + semis) % 12 + 12 * mode).replace(" ", "_")


def transpose(text, semis, piano_range=PIANO_RANGE):
    """Shift every pitch and the key by `semis`. Returns None if any note leaves the piano range."""
    lo, hi = piano_range
    lines = []
    for line in text.splitlines():
        toks = line.split()
        if toks and toks[0] in HANDS:
            out = []
            for tok in toks:
                p = parse_pit(tok)
                if p is None:
                    out.append(tok)
                    continue
                p2 = p + semis
                if not lo <= p2 <= hi:
                    return None
                out.append(pit(p2))
            line = " ".join(out)
        elif toks and toks[0] == "<|meta|>":
            line = re.sub(r"key=(\S+)", lambda m: f"key={transpose_key(m[1], semis)}", line)
        lines.append(line)
    return "\n".join(lines)
