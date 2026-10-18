"""MIDI <-> text codec for score-aligned solo piano MIDI.

Format (each musical field is one vocabulary token; see vocab.py):

    <|piece_start|>
    <|meta|> key=F_minor ts=12/8 bars=262 spb=12 tempo=126
    <|plan|> Exposition:1-65 Development:66-135 ...      (optional; ignored by decode)
    <|bar|> 1 <|pos_0|> <|q_126|> <|ch_I|> <|rhythm_new|> <|phrase|> ts=12/8
    <|mel|> <|v_6|> <|t_0|> <|sd_1|> <|oct_5|> <|d_6|>
    <|rh|> <|v_5|> <|t_6|> <|sd_1|> <|oct_4|> <|sd_3|> <|oct_4|> <|d_6|>
    <|lh|> <|v_4|> <|t_0|> <|sd_1|> <|oct_2|> <|sd_5|> <|oct_2|> <|d_36|>
    <|ped|> <|t_0|> <|ped_on|> <|t_18|> <|ped_off|>
    ...
    <|piece_end|>

Notes are scale degrees of the header key (`<|sd_1|>`..`<|sd_7|>` plus `<|oct_N|>`),
with `<|acc_N|>` only when the note leaves the scale. The top right-hand note at
each attack is `<|mel|>`; the other chord tones stay on `<|rh|>`. `<|ch_|>` is the
bar's triad and is ignored at playback. Tempo is an 8% bin, written only when the
bin changes; other bars are `<|tempo_hold|>`. `<|rhythm_same|>` means this bar's
attacks fall on the same eighth-note beats as the previous bar.

`ts=` appears only when the meter changes, and `len=` only when the bar is irregular.
Rubato within a bar is dropped. Onsets at or past the bar length are clamped on decode.
"""

import bisect
import re
from collections import defaultdict

import mido
import numpy as np
import pretty_midi

from .vocab import (
    PHRASE_BARS,
    chord_token,
    d_on,
    degree_pitch,
    parse_pit,
    parse_tagged_int,
    pitch_tokens,
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

    key = _key_name(pm)
    # Opening tempo is also ordinary text, like the key. Later bars only restate
    # it when the 8% bin changes.
    opening_bpm = 120
    lines = ["<|piece_start|>"]
    prev_ts = None
    prev_bin = None
    prev_pattern = None
    body = []
    for b in range(n_bars):
        t0, t1 = bar_ticks[b], bar_ticks[b + 1]
        ts = ts_changes[max(bisect.bisect_right(ts_ticks, t0) - 1, 0)][1]
        seconds = pm.tick_to_time(t1) - pm.tick_to_time(t0)
        bpm = round(60 * (t1 - t0) / tpb / seconds) if seconds > 0 else 120
        if b == 0:
            opening_bpm = bpm
        bin_ = parse_tagged_int(tempo(bpm), "q")
        pattern = _beat_pattern(events, b, spb)
        rhythm = "<|rhythm_same|>" if prev_pattern and pattern == prev_pattern else "<|rhythm_new|>"
        prev_pattern = pattern
        pitches = [(p, dur) for h in HANDS for (on, dur, _vel), ps in events[h][b].items() for p in ps]
        fields = ["<|bar|>", str(b + 1), pos(b / n_bars)]
        if prev_bin is None or bin_ != prev_bin:
            fields.append(tempo(bpm))
            prev_bin = bin_
        else:
            fields.append("<|tempo_hold|>")
        fields.append(chord_token(key, pitches))
        fields.append(rhythm)
        if b % PHRASE_BARS == 0:
            fields.append("<|phrase|>")
        if ts != prev_ts:
            fields.append(f"ts={ts}")
        prev_ts = ts
        length = bar_steps[b + 1] - bar_steps[b]
        if length != _ts_steps(ts, spb) and b < n_bars - 1:
            fields.append(f"len={length}")
        body.append(" ".join(fields))
        mel, rh = _split_melody(sorted(events["<|rh|>"][b].items()))
        body.append(_hand_line("<|mel|>", mel, key))
        body.append(_hand_line("<|rh|>", rh, key))
        body.append(_hand_line("<|lh|>", sorted(events["<|lh|>"][b].items()), key))
        if ped_by_bar[b]:
            ped = ["<|ped|>"]
            for on, state in ped_by_bar[b]:
                ped += [t_on(on), "<|ped_on|>" if state else "<|ped_off|>"]
            body.append(" ".join(ped))
    lines.append(
        f"<|meta|> key={key} ts={ts_changes[0][1]} bars={n_bars} spb={spb} tempo={tempo(opening_bpm)[4:-2]}"
    )
    lines.extend(body)
    lines.append("<|piece_end|>")
    return "\n".join(lines)


def _beat_pattern(events, bar, spb):
    """Eighth-note bins of every attack. A one-step nudge stays in the same bin."""
    grid = max(1, spb // 2)
    ons = []
    for h in HANDS:
        for (on, _dur, _vel), pitches in events[h][bar].items():
            if pitches:
                ons.append(on // grid)
    return tuple(sorted(ons))


def _split_melody(grouped):
    """Highest pitch at each right-hand attack becomes the melody; the rest stay accompaniment."""
    mel, rh = [], []
    for key, pitches in grouped:
        if not pitches:
            continue
        top = max(pitches)
        mel.append((key, [top]))
        rest = [p for p in pitches if p != top]
        rest += [top] * (pitches.count(top) - 1)
        if rest:
            rh.append((key, rest))
    return mel, rh


def _hand_line(tag, grouped, key):
    toks, cur_vel = [tag], None
    for (on, dur, vel_bin), pitches in grouped:
        if vel_bin != cur_vel:
            toks.append(vel(vel_bin))
            cur_vel = vel_bin
        toks.append(t_on(on))
        for p in pitches:
            toks.extend(pitch_tokens(key, p))
        toks.append(d_on(dur))
    return " ".join(toks)


def parse_hand_notes(rest, key, bar_len=None, default_vel=4):
    """(onset, duration, pitch, velocity bin). Onsets past the bar are clamped."""
    events, onset, pitches = [], None, []
    acc = sd = None
    vel_bin = default_vel
    for tok in rest:
        if tok.startswith("<|v_"):
            vel_bin = parse_tagged_int(tok, "v") if parse_tagged_int(tok, "v") is not None else vel_bin
        elif tok.startswith("<|t_"):
            onset = parse_tagged_int(tok, "t") or 0
            if bar_len and onset >= bar_len:
                onset = bar_len - 1
            pitches, acc, sd = [], 0, None
        elif tok.startswith("<|acc_"):
            acc = parse_tagged_int(tok, "acc") or 0
        elif tok.startswith("<|sd_"):
            sd = parse_tagged_int(tok, "sd")
        elif tok.startswith("<|oct_") and sd:
            pitches.append((degree_pitch(key, sd, parse_tagged_int(tok, "oct") or 4, acc or 0), vel_bin))
            sd, acc = None, 0
        elif tok.startswith("<|pit_"):
            p = parse_pit(tok)
            if p is not None:
                pitches.append((p, vel_bin))
        elif tok.startswith("<|d_") and onset is not None:
            dur = max(1, parse_tagged_int(tok, "d") or 1)
            for p, v in pitches:
                if p is not None and 0 <= p <= 127:
                    events.append((onset, dur, p, v))
            pitches = []
    return events


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
            # <|tempo_hold|> keeps the previous bar's tempo. A new <|q_N|> is a new bin.
            tempo_events.append((bar_start, bpm))
        elif tag in ("<|mel|>",) + HANDS and in_bar:
            # Melody is the top of the right hand; playback merges it back onto that track.
            dest = "<|rh|>" if tag == "<|mel|>" else tag
            for onset, dur, pitch, vel_bin in parse_hand_notes(rest, key, bar_len, vel_bins // 2):
                velocity = min(127, vel_bin * (128 // vel_bins) + 128 // vel_bins // 2)
                notes[dest].append((bar_start + onset, dur, pitch, velocity))
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
    """Shift the key by `semis`. Degrees stay put, so the sounding pitches move with the key.

    Returns None if any sounding pitch would leave the piano. Legacy `<|pit_|>` scores
    are rewritten note by note.
    """
    if not semis:
        return text
    lo, hi = piano_range
    meta = next((ln for ln in text.splitlines() if ln.startswith("<|meta|>")), "")
    key = dict(_FIELD.findall(meta)).get("key")
    sounding = []
    for line in text.splitlines():
        toks = line.split()
        if toks and toks[0] in ("<|mel|>",) + HANDS:
            sounding += [p for _o, _d, p, _v in parse_hand_notes(toks[1:], key)]
    if any(not lo <= p + semis <= hi for p in sounding):
        return None
    # Degree spellings are relative to the key, so only the key name changes.
    # A leftover absolute pitch token has to move explicitly.
    from .vocab import pit
    lines = []
    for line in text.splitlines():
        toks = line.split()
        if toks and toks[0] in ("<|mel|>",) + HANDS and any(t.startswith("<|pit_") for t in toks):
            out = []
            for tok in toks:
                p = parse_pit(tok)
                if p is None:
                    out.append(tok)
                else:
                    out.append(pit(p + semis))
            line = " ".join(out)
        elif toks and toks[0] == "<|meta|>":
            line = re.sub(r"key=(\S+)", lambda m: f"key={transpose_key(m[1], semis)}", line)
        lines.append(line)
    return "\n".join(lines)
