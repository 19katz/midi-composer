"""Chunked generation: header + opening bars, then continuations until the piece ends."""

import inspect

from .dataset import approx_tokens, make_prompt, meta_fields, split_piece


class PieceStop:
    """Stop a chunk once it has written the bars it was asked for, or `<|piece_end|>`.

    The token that opens the next bar is what shows the previous bar is finished,
    so `max_bars=40` stops on the 41st `<|bar|>`, not the 40th. Tokens in the
    prompt are not counted.
    """

    def __init__(self, prompt_len, bar_id, end_id, max_bars):
        self.seen = int(prompt_len)
        self.bar_id = bar_id
        self.end_id = end_id
        self.max_bars = None if max_bars is None else int(max_bars)
        self.bars = 0

    def __call__(self, input_ids, scores=None, **kwargs):
        row = input_ids[0]
        new = row[self.seen:]
        ids = new.tolist() if hasattr(new, "tolist") else list(new)
        self.seen = row.shape[0] if hasattr(row, "shape") else len(row)
        for i in ids:
            if self.end_id is not None and i == self.end_id:
                return True
            if i == self.bar_id:
                self.bars += 1
                if self.max_bars is not None and self.bars > self.max_bars:
                    return True
        return False


def _call_generate(generate_fn, messages, max_bars):
    """Pass the remaining bar count when the sampler accepts it.

    Older one-argument callables, including the tests, keep working.
    """
    try:
        params = inspect.signature(generate_fn).parameters
    except (TypeError, ValueError):
        return generate_fn(messages)
    var = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if "max_bars" in params or var:
        return generate_fn(messages, max_bars=max_bars)
    return generate_fn(messages)


def _complete_bars(text):
    """Bars from model output, dropping a trailing bar that is only a header line."""
    _, bars = split_piece(text)
    # A tok2 bar is header + melody + rh + lh. One line means the bar never started.
    if bars and len(bars[-1].splitlines()) < 2 and "<|piece_end|>" not in text:
        bars = bars[:-1]
    return bars


def _brief_bars(description):
    """Bar count named in the brief, e.g. 'about 59 bars'. None when the brief doesn't say."""
    import re
    m = re.search(r"about\s+(\d+)\s+bars", description or "", re.I)
    return int(m.group(1)) if m else None


def compose(generate_fn, description, key=None, meter=None, bars=None, max_chunks=40, context_budget=1200,
            max_context_bars=16, measure=approx_tokens, on_chunk=None):
    """`generate_fn(messages) -> str` produces one assistant reply. Returns the full score text.

    Length comes from `bars`, or from "about N bars" in the brief. The model's own
    header is rewritten to that count, and generation stops there. A Chopin prelude
    ran to 78 bars because the header, not the brief, set the length.
    """
    planned = bars or _brief_bars(description)
    first = _call_generate(generate_fn, make_prompt(description, key, meter), planned)
    header = [l for l in first.splitlines() if l.startswith(("<|meta|>", "<|plan|>"))]
    if not header:
        header = [f"<|meta|> key={key or 'C_Major'} ts={meter or '4/4'} bars={planned or 64} spb=12"]
    if planned:
        import re
        header[0] = re.sub(r"bars=\d+", f"bars={planned}", header[0])
        if "bars=" not in header[0]:
            header[0] += f" bars={planned}"
    total = planned or int(meta_fields(header[0]).get("bars", 0) or 0)
    bars = _complete_bars(first)
    done = "<|piece_end|>" in first
    if on_chunk:
        on_chunk(len(bars), total)

    for _ in range(max_chunks - 1):
        if done or (total and len(bars) >= total):
            break
        ctx, used = [], 0
        for b in reversed(bars[-max_context_bars:]):
            if used + measure(b) > context_budget:
                break
            ctx.insert(0, b)
            used += measure(b)
        remaining = (total - len(bars)) if total else None
        out = _call_generate(
            generate_fn, make_prompt(description, key, meter, header, ctx, len(bars) + 1), remaining)
        new = _complete_bars(out)
        if not new:
            break
        bars += new
        done = "<|piece_end|>" in out
        if on_chunk:
            on_chunk(len(bars), total)

    if total:
        bars = bars[:total]
    return "\n".join(["<|piece_start|>", *header, *bars, "<|piece_end|>"])
