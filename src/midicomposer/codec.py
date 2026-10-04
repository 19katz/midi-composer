"""MIDI <-> text codec for score-aligned solo piano MIDI.

Format (one line per tag):

    <|piece_start|>
    <|meta|> key=F_minor ts=12/8 bars=262 spb=12
    <|bar|> n=1 pos=0.00 ts=12/8 tempo=126
    <|rh|> 0:C5:v6:6 6:Ab4.C5:v5:6
    <|lh|> 0:F2.C3:v4:36
    ...
    <|piece_end|>

Note events are `onset:pitches:velocity_bin:duration`, with onset relative to the bar
and onset/duration measured in grid steps (`spb` steps per quarter note). Simultaneous
notes sharing onset, duration, and velocity are joined with `.`. `ts=` on a bar line
appears only when the time signature changes, and `len=` only when the bar's length in
steps differs from what its time signature implies (cadenzas, irregular bars). `tempo` is the bar's average quarter-note
BPM, so bar start times survive decoding but rubato within a bar does not.
"""

import bisect
import re
from collections import defaultdict

import mido
import pretty_midi

SPB = 12
VEL_BINS = 8
TICKS_PER_STEP = 40

_MIDO_KEYS = [
    "C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B",
    "Cm", "C#m", "Dm", "Ebm", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "Bbm", "Bm",
]

HANDS = ("rh", "lh")
TRACK_NAMES = {"rh": "Piano right", "lh": "Piano left"}


def _hand(inst):
    return "lh" if "left" in inst.name.lower() else "rh"


def _key_name(pm):
    if not pm.key_signature_changes:
        return "unknown"
    return pretty_midi.key_number_to_key_name(pm.key_signature_changes[0].key_number).replace(" ", "_")


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
        ts_field = f" ts={ts}" if ts != prev_ts else ""
        prev_ts = ts
        length = bar_steps[b + 1] - bar_steps[b]
        len_field = f" len={length}" if length != _ts_steps(ts, spb) and b < n_bars - 1 else ""
        lines.append(f"<|bar|> n={b + 1} pos={b / n_bars:.2f}{ts_field}{len_field} tempo={tempo}")
        for h in HANDS:
            toks = []
            for (on, dur, vel), pitches in sorted(events[h][b].items()):
                names = ".".join(pretty_midi.note_number_to_name(p) for p in sorted(pitches))
                toks.append(f"{on}:{names}:v{vel}:{dur}")
            lines.append(f"<|{h}|> " + " ".join(toks))
    lines.append("<|piece_end|>")
    return "\n".join(lines)


_FIELD = re.compile(r"(\w+)=(\S+)")
_NOTE = re.compile(r"^(\d+):([A-G][#b]?-?\d(?:\.[A-G][#b]?-?\d)*):v(\d+):(\d+)$")


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
        elif tag == "<|bar|>":
            if in_bar:
                bar_start += bar_len
            in_bar = True
            if "ts" in fields and fields["ts"] != (ts_events[-1][1] if ts_events else None):
                ts = fields["ts"]
                ts_events.append((bar_start, ts))
            elif not ts_events:
                ts_events.append((bar_start, ts))
            bar_len = int(fields["len"]) if "len" in fields else _ts_steps(ts, spb)
            tempo = float(fields.get("tempo", tempo))
            tempo_events.append((bar_start, tempo))
        elif tag in ("<|rh|>", "<|lh|>") and in_bar:
            hand = tag[2:4]
            for tok in rest.split():
                m = _NOTE.match(tok)
                if not m:
                    continue
                on, names, vel, dur = int(m[1]), m[2], int(m[3]), int(m[4])
                velocity = min(127, vel * (128 // vel_bins) + 128 // vel_bins // 2)
                for name in names.split("."):
                    try:
                        pitch = pretty_midi.note_name_to_number(name)
                    except Exception:
                        continue
                    if 0 <= pitch <= 127:
                        notes[hand].append((bar_start + on, max(1, dur), pitch, velocity))

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
