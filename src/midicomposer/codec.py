"""MIDI <-> text codec for score-aligned solo piano MIDI.

Format:

    <|piece_start|>
    <|meta|> key=F_Minor ts=12/8 bars=262 spb=12
    <|plan|> Exposition:1-65 Development:66-135 ...      (optional; ignored by decode)
    bar 1 @0.00 q126 ts=12/8
    R v6 0:C5:6 v5 6:Ab4.C5:6
    L v4 0:F2.C3:36
    ...
    <|piece_end|>

Bar lines: bar number, `@` fractional position in the piece, `q` average quarter-note
BPM over the bar (bar start times survive decoding; rubato within a bar does not),
`ts=` only when the time signature changes, and `len=` only when the bar's length in
grid steps differs from what its time signature implies (cadenzas, irregular bars).

Hand lines (`R`, `L`): note events `onset:pitches:duration`, onset relative to the bar,
both in grid steps (`spb` steps per quarter note). Simultaneous notes sharing onset,
duration, and velocity are joined with `.`. A `vN` token sets the velocity bin for the
events that follow it; every hand line that has notes starts with one.
"""

import bisect
import re
from collections import defaultdict

import mido
import numpy as np
import pretty_midi

SPB = 12
VEL_BINS = 8
TICKS_PER_STEP = 40
PIANO_RANGE = (21, 108)

_MIDO_KEYS = [
    "C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B",
    "Cm", "C#m", "Dm", "Ebm", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "Bbm", "Bm",
]

HANDS = ("R", "L")
TRACK_NAMES = {"R": "Piano right", "L": "Piano left"}


def _hand(inst):
    return "L" if "left" in inst.name.lower() else "R"


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
    major = marked
    minor = (major + 9) % 12
    notes = [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]
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


def encode(pm, spb=SPB, vel_bins=VEL_BINS):
    tpb = pm.resolution
    step = tpb / spb
    notes = [(inst, n) for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    end_tick = max(pm.time_to_tick(n.end) for _, n in notes)

    bar_ticks = sorted({pm.time_to_tick(t) for t in pm.get_downbeats()})
    bar_ticks = [t for t in bar_ticks if t < end_tick] or [0]
    n_bars = len(bar_ticks)
    bar_ticks.append(max(end_tick, bar_ticks[-1] + 1))
    bar_steps = [round(t / step) for t in bar_ticks]

    ts_changes = sorted(
        (pm.time_to_tick(ts.time), f"{ts.numerator}/{ts.denominator}") for ts in pm.time_signature_changes
    ) or [(0, "4/4")]
    ts_ticks = [t for t, _ in ts_changes]

    events = {h: [defaultdict(set) for _ in range(n_bars)] for h in HANDS}
    for inst, n in notes:
        s = round(pm.time_to_tick(n.start) / step)
        dur = max(1, round(pm.time_to_tick(n.end) / step) - s)
        b = min(max(bisect.bisect_right(bar_steps, s) - 1, 0), n_bars - 1)
        vel = min(vel_bins - 1, n.velocity * vel_bins // 128)
        events[_hand(inst)][b][(s - bar_steps[b], dur, vel)].add(n.pitch)

    lines = ["<|piece_start|>", f"<|meta|> key={_key_name(pm)} ts={ts_changes[0][1]} bars={n_bars} spb={spb}"]
    prev_ts = None
    for b in range(n_bars):
        t0, t1 = bar_ticks[b], bar_ticks[b + 1]
        ts = ts_changes[max(bisect.bisect_right(ts_ticks, t0) - 1, 0)][1]
        seconds = pm.tick_to_time(t1) - pm.tick_to_time(t0)
        tempo = round(60 * (t1 - t0) / tpb / seconds) if seconds > 0 else 120
        fields = [f"bar {b + 1}", f"@{b / n_bars:.2f}", f"q{tempo}"]
        if ts != prev_ts:
            fields.append(f"ts={ts}")
        prev_ts = ts
        length = bar_steps[b + 1] - bar_steps[b]
        if length != _ts_steps(ts, spb) and b < n_bars - 1:
            fields.append(f"len={length}")
        lines.append(" ".join(fields))
        for h in HANDS:
            toks, cur_vel = [h], None
            for (on, dur, vel), pitches in sorted(events[h][b].items()):
                if vel != cur_vel:
                    toks.append(f"v{vel}")
                    cur_vel = vel
                names = ".".join(pretty_midi.note_number_to_name(p) for p in sorted(pitches))
                toks.append(f"{on}:{names}:{dur}")
            lines.append(" ".join(toks))
    lines.append("<|piece_end|>")
    return "\n".join(lines)


_FIELD = re.compile(r"(\w+)=(\S+)")
_PITCH = r"[A-G][#b]?-?\d"
_NOTE = re.compile(rf"^(\d+):({_PITCH}(?:\.{_PITCH})*):(\d+)$")


def decode(text, vel_bins=VEL_BINS):
    """Parse codec text into a mido.MidiFile. Malformed lines and tokens are skipped."""
    spb, ts, tempo, key = SPB, "4/4", 120, None
    bar_start, bar_len, in_bar = 0, None, False
    tempo_events, ts_events = [], []
    notes = {h: [] for h in HANDS}

    for line in text.splitlines():
        tag, _, rest = line.strip().partition(" ")
        fields = dict(_FIELD.findall(rest))
        if tag == "<|meta|>":
            spb = int(fields.get("spb", spb))
            ts = fields.get("ts", ts)
            key = fields.get("key")
        elif tag == "bar":
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
            q = re.search(r"\bq(\d+(?:\.\d+)?)\b", rest)
            if q and float(q[1]) > 0:
                tempo = float(q[1])
            tempo_events.append((bar_start, tempo))
        elif tag in HANDS and in_bar:
            vel = vel_bins // 2
            for tok in rest.split():
                if re.fullmatch(r"v\d+", tok):
                    vel = min(int(tok[1:]), vel_bins - 1)
                    continue
                m = _NOTE.match(tok)
                if not m:
                    continue
                on, names, dur = int(m[1]), m[2], int(m[3])
                velocity = min(127, vel * (128 // vel_bins) + 128 // vel_bins // 2)
                for name in names.split("."):
                    try:
                        pitch = pretty_midi.note_name_to_number(name)
                    except Exception:
                        continue
                    if 0 <= pitch <= 127:
                        notes[tag].append((bar_start + on, max(1, dur), pitch, velocity))

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
        for on, dur, pitch, vel in notes[h]:
            evs.append((on * TICKS_PER_STEP, 1, mido.Message("note_on", note=pitch, velocity=vel)))
            evs.append(((on + dur) * TICKS_PER_STEP, 0, mido.Message("note_off", note=pitch, velocity=0)))
        track = _to_track(evs)
        track.insert(0, mido.MetaMessage("track_name", name=TRACK_NAMES[h]))
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
        tag = line.split(" ", 1)[0]
        if tag in HANDS:
            toks = []
            for tok in line.split(" "):
                m = _NOTE.match(tok)
                if m:
                    pitches = [pretty_midi.note_name_to_number(p) + semis for p in m[2].split(".")]
                    if not all(lo <= p <= hi for p in pitches):
                        return None
                    tok = f"{m[1]}:{'.'.join(pretty_midi.note_number_to_name(p) for p in pitches)}:{m[3]}"
                toks.append(tok)
            line = " ".join(toks)
        elif tag == "<|meta|>":
            line = re.sub(r"key=(\S+)", lambda m: f"key={transpose_key(m[1], semis)}", line)
        lines.append(line)
    return "\n".join(lines)
