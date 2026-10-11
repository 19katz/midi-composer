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
POS_BINS = 100
PIANO_LO, PIANO_HI = 21, 108

STRUCT = (
    "<|piece_start|>",
    "<|piece_end|>",
    "<|meta|>",
    "<|plan|>",
    "<|bar|>",
    "<|rh|>",
    "<|lh|>",
    "<|ped|>",
    "<|ped_on|>",
    "<|ped_off|>",
)

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


def tempo(bpm):
    return f"<|q_{max(TEMPO_MIN, min(TEMPO_MAX, int(round(bpm))))}|>"


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
    """Stable list of every added token. Order is part of the public interface."""
    tokens = list(STRUCT)
    tokens += [vel(i) for i in range(8)]
    tokens += [t_on(i) for i in range(MAX_ONSET + 1)]
    tokens += [d_on(i) for i in range(1, MAX_DUR + 1)]
    tokens += [tempo(i) for i in range(TEMPO_MIN, TEMPO_MAX + 1)]
    tokens += [pos(i / POS_BINS) for i in range(POS_BINS + 1)]
    tokens += [pit(n) for n in range(PIANO_LO, PIANO_HI + 1)]
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
