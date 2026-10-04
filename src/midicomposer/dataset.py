"""Turn encoded pieces + descriptions into prompt/completion training windows.

Each piece is cut into windows. The first window's completion is the header (piece start,
meta, plan) plus the opening bars; later windows give the header and the most recent bars
as context and ask for the next bars. The same `make_prompt` is used at generation time.
"""

import random
import re

from .codec import transpose
from .describe import VARIANTS

SYSTEM = "You are a composer of classical solo piano music. You write scores in a compact text notation."


def approx_tokens(text):
    return len(text) / 1.25


def split_piece(text):
    """Return (meta_line, bars) where each bar is its 'bar' line plus its hand lines."""
    meta, bars, cur = None, [], []
    for line in text.splitlines():
        if line.startswith("<|meta|>"):
            meta = line
        elif line.startswith("bar "):
            if cur:
                bars.append("\n".join(cur))
            cur = [line]
        elif line[:2] in ("R ", "L ") or line in ("R", "L"):
            if cur:
                cur.append(line)
    if cur:
        bars.append("\n".join(cur))
    return meta, bars


def plan_line(sections):
    if not sections:
        return None
    parts = []
    for s in sections:
        try:
            a, b = s["bars"]
            parts.append(f"{re.sub(r'[^A-Za-z0-9]+', '_', s['name']).strip('_')}:{int(a)}-{int(b)}")
        except (KeyError, TypeError, ValueError):
            continue
    return "<|plan|> " + " ".join(parts) if parts else None


def meta_fields(meta):
    return dict(re.findall(r"(\w+)=(\S+)", meta or ""))


def make_prompt(description, key=None, meter=None, header=None, context_bars=None, next_bar=None):
    """Chat messages for either the opening window (header=None) or a continuation."""
    lines = ["Compose a classical solo piano piece."]
    if key:
        lines.append(f"Key: {key.replace('_', ' ')}")
    if meter:
        lines.append(f"Meter: {meter}")
    lines += ["", description.strip()]
    if header is not None:
        lines += ["", f"Continue the score from bar {next_bar}. Header and most recent bars:", *header]
        lines += context_bars or []
    else:
        lines += ["", "Write the score header and the opening bars."]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(lines)}]


def windows(text, sections=None, measure=approx_tokens, target_budget=3000, context_budget=1200,
            max_target_bars=32, max_context_bars=8):
    """Yield (header, context_bars, target_text, next_bar_number) tuples covering the whole piece."""
    meta, bars = split_piece(text)
    plan = plan_line(sections)
    header = [meta] + ([plan] if plan else [])
    start = 0
    while start < len(bars):
        end, used = start, 0
        while end < len(bars) and end - start < max_target_bars:
            cost = measure(bars[end])
            if end > start and used + cost > target_budget:
                break
            used += cost
            end += 1
        ctx, used = [], 0
        for b in reversed(bars[max(0, start - max_context_bars):start]):
            cost = measure(b)
            if used + cost > context_budget:
                break
            ctx.insert(0, b)
            used += cost
        target = bars[start:end]
        if start == 0:
            target = ["<|piece_start|>", *header, *target]
        if end == len(bars):
            target = [*target, "<|piece_end|>"]
        yield (header if start else None), ctx, "\n".join(target), start + 1
        start = end


def build_examples(text, desc, semitones=(0,), key_dropout=0.3, rng=None, **window_kw):
    """Prompt/completion examples for one piece across transpositions."""
    rng = rng or random.Random(0)
    out = []
    for semis in semitones:
        t = transpose(text, semis) if semis else text
        if t is None:
            continue
        m = meta_fields(t.splitlines()[1])
        for header, ctx, target, next_bar in windows(t, desc.get("sections"), **window_kw):
            variant = rng.choice([v for v in VARIANTS if desc.get(v)])
            key = m.get("key") if rng.random() > key_dropout else None
            meter = m.get("ts") if rng.random() > key_dropout else None
            out.append({
                "prompt": make_prompt(desc[variant], key, meter, header, ctx, next_bar),
                "completion": [{"role": "assistant", "content": target}],
                "file": desc["file"],
                "semis": semis,
            })
    return out


def build_split(texts, descs, files, semitones=(0,), per_piece=None, seed=0, **kw):
    """Examples for `files`, given {file: encoded text} and {file: description record}.
    With `per_piece`, each piece gets the original key plus `per_piece - 1` random shifts."""
    rng = random.Random(seed)
    out = []
    for f in files:
        if f not in texts or f not in descs:
            continue
        shifts = list(semitones)
        if per_piece and per_piece < len(shifts):
            others = [s for s in shifts if s != 0]
            shifts = [0] + rng.sample(others, per_piece - 1)
        out += build_examples(texts[f], descs[f], shifts, rng=rng, **kw)
    rng.shuffle(out)
    return out


def work_id(filename):
    """Group movements of one work: 'appass_1.mid' -> 'appass', 'mz_570_3.mid' -> 'mz_570'."""
    stem = filename.rsplit(".", 1)[0]
    return re.sub(r"_\d+$", "", stem)


def split_by_work(filenames, test_frac=0.1, seed=0):
    works = sorted({work_id(f) for f in filenames})
    random.Random(seed).shuffle(works)
    test_works = set(works[: max(1, round(len(works) * test_frac))])
    train = [f for f in filenames if work_id(f) not in test_works]
    test = [f for f in filenames if work_id(f) in test_works]
    return train, test
