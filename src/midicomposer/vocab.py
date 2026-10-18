"""Dedicated music tokens added to the base LLM vocabulary.

Each musical field is one token, spelled `<|...|>` so it cannot appear inside an
English brief. Bar numbers, keys, and meters stay as ordinary text: the pretrained
model already counts and already knows key names.

Call `music_tokens()` for the list to pass to Unsloth's `add_new_tokens` (before
`get_peft_model`), with `method="interpolation"` so each new embedding starts as a
blend of the mean embedding and the mean of the subword pieces it replaces.
"""

import re

import pretty_midi

MAX_ONSET = 192
MAX_DUR = 256
TEMPO_MIN, TEMPO_MAX = 20, 260
# One tempo token per ~8% step. 119 vs 120 is not a musical difference, and 241
# integer tokens were mostly rare. Crossing into the next bin is the change we keep.
TEMPO_STEP = 1.08
POS_BINS = 100
PIANO_LO, PIANO_HI = 21, 108
# Phrase length. Tempo can change on any bar (when it crosses a bin); the phrase
# mark is only a boundary the model can attend to.
PHRASE_BARS = 4

STRUCT = (
    "<|piece_start|>",
    "<|piece_end|>",
    "<|meta|>",
    "<|plan|>",
    "<|bar|>",
    "<|phrase|>",
    "<|mel|>",
    "<|rh|>",
    "<|lh|>",
    "<|ped|>",
    "<|ped_on|>",
    "<|ped_off|>",
    "<|tempo_hold|>",
    "<|rhythm_same|>",
    "<|rhythm_new|>",
)

# Diatonic triads in the key, plus a seventh when that pitch is actually present.
ROMANS = ("I", "ii", "iii", "IV", "V", "vi", "vii")
_MAJOR_IV = (0, 2, 4, 5, 7, 9, 11)
_MINOR_IV = (0, 2, 3, 5, 7, 8, 10)

_TAG = re.compile(r"<\|[^|]+\|>")


def pitch_name(midi_n):
    """Canonical pitch spelling: sharps as s (C#4 -> Cs4)."""
    return pretty_midi.note_number_to_name(int(midi_n)).replace("#", "s")


def parse_pitch_name(name):
    return pretty_midi.note_name_to_number(re.sub(r"([A-G])s", r"\1#", name))


def pit(midi_n):
    return f"<|pit_{pitch_name(midi_n)}|>"


def t_on(step):
    return f"<|t_{max(0, min(MAX_ONSET, int(step)))}|>"


def d_on(step):
    return f"<|d_{max(1, min(MAX_DUR, int(step)))}|>"


def vel(bin_):
    return f"<|v_{max(0, min(7, int(bin_)))}|>"


def tempo_centers():
    """Bin centers from TEMPO_MIN to TEMPO_MAX, each ~8% faster than the last."""
    import math
    centers, t = [], float(TEMPO_MIN)
    while t <= TEMPO_MAX + 1e-6:
        centers.append(int(round(t)))
        t *= TEMPO_STEP
    # Rounding can collide (two steps land on the same integer). Keep the first.
    out = []
    for c in centers:
        if not out or c != out[-1]:
            out.append(c)
    return out


def tempo_bin(bpm):
    """Nearest bin center on a log scale, so a sub-8% wobble stays in the bin."""
    import math
    bpm = max(TEMPO_MIN, min(TEMPO_MAX, float(bpm)))
    return min(tempo_centers(), key=lambda c: abs(math.log(bpm) - math.log(c)))


def tempo(bpm):
    return f"<|q_{tempo_bin(bpm)}|>"


def scale_pcs(key):
    """(tonic pitch class, 7 scale intervals). Unknown keys fall back to C major."""
    try:
        number = pretty_midi.key_name_to_key_number((key or "C major").replace("_", " "))
    except Exception:
        number = 0
    intervals = _MINOR_IV if number >= 12 else _MAJOR_IV
    return number % 12, intervals


def pitch_tokens(key, midi_n):
    """Canonical degree/octave/accidental spelling of one MIDI pitch.

    Prefer no accidental, then the smaller one, so each sounding pitch has one
    written form. `<|acc_1|> <|sd_1|>` and `<|acc_-1|> <|sd_2|>` can be the same
    key; only the first of those ranks is emitted.
    """
    root, intervals = scale_pcs(key)
    target = int(midi_n) % 12
    octave = int(midi_n) // 12 - 1
    best = None
    for sd, iv in enumerate(intervals, 1):
        for acc in (0, 1, -1, 2, -2):
            if (root + iv + acc) % 12 == target:
                rank = (abs(acc), 0 if acc >= 0 else 1, sd)
                if best is None or rank < best[0]:
                    best = (rank, sd, acc)
                break
    _rank, sd, acc = best
    toks = [f"<|acc_{acc}|>"] if acc else []
    return toks + [f"<|sd_{sd}|>", f"<|oct_{octave}|>"]


def degree_pitch(key, sd, octave, acc=0):
    """MIDI pitch for a degree spelling. Inverse of `pitch_tokens` when spelling is canonical."""
    root, intervals = scale_pcs(key)
    pc = (root + intervals[int(sd) - 1] + int(acc)) % 12
    return pc + (int(octave) + 1) * 12


def chord_token(key, pitch_durs):
    """`<|ch_V|>` or `<|ch_V7|>` for the diatonic triad that covers the most duration."""
    if not pitch_durs:
        return "<|ch_none|>"
    root, intervals = scale_pcs(key)
    best = None
    for i, roman in enumerate(ROMANS):
        triad = {(root + intervals[(i + k) % 7]) % 12 for k in (0, 2, 4)}
        seventh = (root + intervals[(i + 6) % 7]) % 12
        score = has7 = 0
        for pitch, dur in pitch_durs:
            pc = int(pitch) % 12
            if pc in triad:
                score += dur
            elif pc == seventh:
                has7 += dur
        rank = (score, has7)
        if best is None or rank > best[0]:
            best = (rank, roman, has7 >= max(1, score) * 0.25 and has7 > 0)
    _rank, roman, use7 = best
    if _rank[0] <= 0:
        return "<|ch_none|>"
    return f"<|ch_{roman}7|>" if use7 else f"<|ch_{roman}|>"


def chord_pcs(key, name):
    """Pitch classes allowed by a chord token, or None when the bar has no chord."""
    if not name or name == "none":
        return None
    use7 = name.endswith("7")
    roman = name[:-1] if use7 else name
    if roman not in ROMANS:
        return None
    i = ROMANS.index(roman)
    root, intervals = scale_pcs(key)
    pcs = {(root + intervals[(i + k) % 7]) % 12 for k in (0, 2, 4)}
    if use7:
        pcs.add((root + intervals[(i + 6) % 7]) % 12)
    return pcs


def pos(frac):
    return f"<|pos_{max(0, min(POS_BINS, int(round(frac * POS_BINS))))}|>"


def parse_tagged_int(tok, prefix):
    m = re.fullmatch(rf"<\|{prefix}_(-?\d+)\|>", tok)
    return int(m[1]) if m else None


def parse_pit(tok):
    m = re.fullmatch(r"<\|pit_([A-G][sb]?(?:-?\d+))\|>", tok)
    if not m:
        return None
    try:
        return parse_pitch_name(m[1])
    except Exception:
        return None


def music_tokens():
    """Stable list of every added token. Order is part of the public interface.

    Pitches are scale degrees of the header key, not 88 absolute names. Tempo is
    one token per ~8% bin, not one per BPM. Absolute `<|pit_|>` spellings are
    still parsed by the decoder so older scores can be read, but they are not
    in this list and the model is not trained to emit them.
    """
    tokens = list(STRUCT)
    tokens += [f"<|ch_{r}|>" for r in ROMANS]
    tokens += [f"<|ch_{r}7|>" for r in ROMANS]
    tokens.append("<|ch_none|>")
    tokens += [f"<|sd_{d}|>" for d in range(1, 8)]
    tokens += [f"<|oct_{o}|>" for o in range(9)]
    tokens += [f"<|acc_{a}|>" for a in (-2, -1, 1, 2)]
    tokens += [vel(i) for i in range(8)]
    tokens += [t_on(i) for i in range(MAX_ONSET + 1)]
    tokens += [d_on(i) for i in range(1, MAX_DUR + 1)]
    tokens += [tempo(c) for c in tempo_centers()]
    tokens += [pos(i / POS_BINS) for i in range(POS_BINS + 1)]
    seen, out = set(), []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def approx_tokens(text):
    """Token count after the music vocabulary is added: one per `<|...|>`, one per leftover word."""
    specials = _TAG.findall(text)
    rest = _TAG.sub(" ", text)
    return len(specials) + len(rest.split())


def attach_music_tokens(model, tokenizer, interpolation=0.5):
    """Resize the model/tokenizer and initialize new embeddings. Call before LoRA.

    Uses Unsloth when available (`method="interpolation"`). Otherwise adds the tokens
    without touching embeddings, which is enough for CPU tests.
    """
    tokens = music_tokens()
    try:
        from unsloth import add_new_tokens
    except ImportError:
        existing = set(tokenizer.get_vocab())
        tokenizer.add_tokens([t for t in tokens if t not in existing])
        if model is not None and hasattr(model, "resize_token_embeddings"):
            model.resize_token_embeddings(len(tokenizer))
        return tokenizer
    add_new_tokens(model, tokenizer, new_tokens=tokens, method="interpolation", interpolation=interpolation)
    return tokenizer
