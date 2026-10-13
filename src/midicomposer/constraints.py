"""Sampling masks for a checkpoint that already writes valid score text.

Applied inside `model.generate`, so later tokens are chosen given a legal pitch,
onset, rhythm, and tempo. Three masks:

- pitch: while a pitch token is expected, penalize notes outside the key. After
  `chromatic_budget` chromatic notes in the bar, ban further ones.
- onset: while an onset token is expected, ban steps at or past the bar length.
- rhythm / tempo: ban the duration that would make the right hand an exact copy
  of the previous bar, and ban tempo tokens within `tempo_ratio` of the previous
  bar (the previous tempo itself stays allowed).
"""

import re

from .codec import _ts_steps
from .evaluate import _scale
from .vocab import music_tokens, parse_pit, parse_tagged_int

try:
    from transformers import LogitsProcessor
except ImportError:  # CPU tests, and any environment without the training stack
    class LogitsProcessor:
        pass


_MUSIC = re.compile(r"<\|[^|]+\|>")
_HANDS = ("<|rh|>", "<|lh|>")


class ScoreConstraints(LogitsProcessor):
    def __init__(self, tokenizer, pitch_penalty=8.0, chromatic_budget=2, tempo_ratio=0.08):
        self.tokenizer = tokenizer
        self.pitch_penalty = float(pitch_penalty)
        self.chromatic_budget = int(chromatic_budget)
        self.tempo_ratio = float(tempo_ratio)
        unk = getattr(tokenizer, "unk_token_id", None)
        self.ids = {}
        for tok in music_tokens():
            i = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(i, int) and i >= 0 and i != unk:
                self.ids[tok] = i
        self.onset_ids = {parse_tagged_int(t, "t"): i for t, i in self.ids.items() if t.startswith("<|t_")}
        self.dur_ids = {parse_tagged_int(t, "d"): i for t, i in self.ids.items() if t.startswith("<|d_")}
        self.tempo_ids = {parse_tagged_int(t, "q"): i for t, i in self.ids.items() if t.startswith("<|q_")}
        self.pit_ids = {}
        for tok, i in self.ids.items():
            pitch = parse_pit(tok)
            if pitch is not None:
                self.pit_ids[pitch] = i
        self.reset()

    def reset(self):
        self.seen = 0
        self.raw = ""
        self.header_buf = ""
        self.phase = "header"
        self.hand = None
        self.spb = 12
        self.key = None
        self.scale = None
        self.default_bar_len = _ts_steps("4/4", self.spb)
        self.bar_len = self.default_bar_len
        self.len_override = False
        self.in_bar = False
        self.prev_q = None
        self.cur_q = None
        self.prev_rh = None
        self.cur_rh = []
        self.pending_onset = None
        self.chromatic_count = 0

    def __call__(self, input_ids, scores):
        row = input_ids[0]
        n = row.shape[0]
        new = row[self.seen:n]
        new = new.tolist() if hasattr(new, "tolist") else list(new)
        opening = self.seen == 0
        for i in new:
            self._consume(int(i))
        self.seen = n
        if opening:
            self._finish_prompt()
        return self._apply(scores)

    def _piece(self, token_id):
        tok = self.tokenizer.convert_ids_to_tokens(token_id) or ""
        return tok.replace("▁", " ").replace("Ġ", " ")

    def _consume(self, token_id):
        piece = self._piece(token_id)
        self.raw += piece
        tok = piece.strip()
        if _MUSIC.fullmatch(tok):
            self._music(tok)
        elif self.phase in ("header", "meta"):
            self.header_buf += piece
        self._sync_fields()

    def _music(self, tok):
        if tok == "<|bar|>":
            self._on_bar()
        elif tok == "<|meta|>":
            self.phase = "meta"
            self.header_buf = ""
        elif tok in ("<|rh|>", "<|lh|>", "<|ped|>"):
            self.hand = tok
            self.phase = "onset"
            self.pending_onset = None
        elif tok.startswith("<|v_") or tok in ("<|ped_on|>", "<|ped_off|>"):
            self.phase = "onset"
        elif tok.startswith("<|t_"):
            self.pending_onset = parse_tagged_int(tok, "t")
            if self.hand in _HANDS:
                self.phase = "pitch"
        elif tok.startswith("<|pit_"):
            pitch = parse_pit(tok)
            if pitch is not None and self.scale:
                root, pcs = self.scale
                if (pitch - root) % 12 not in pcs:
                    self.chromatic_count += 1
            self.phase = "pitch"
        elif tok.startswith("<|d_"):
            dur = parse_tagged_int(tok, "d") or 1
            if self.hand == "<|rh|>" and self.pending_onset is not None:
                self.cur_rh.append((self.pending_onset, dur))
            self.pending_onset = None
            self.phase = "onset"
        elif tok.startswith("<|q_"):
            self.cur_q = parse_tagged_int(tok, "q")

    def _on_bar(self):
        if self.in_bar:
            self.prev_rh = tuple(self.cur_rh)
            if self.cur_q:
                self.prev_q = self.cur_q
        self.cur_rh = []
        self.cur_q = None
        self.hand = None
        self.pending_onset = None
        self.chromatic_count = 0
        self.header_buf = ""
        self.len_override = False
        self.bar_len = self.default_bar_len
        self.in_bar = True
        self.phase = "header"

    def _finish_prompt(self):
        """Context bars in the prompt are complete. The next token starts the continuation."""
        if self.in_bar:
            self.prev_rh = tuple(self.cur_rh)
            if self.cur_q:
                self.prev_q = self.cur_q
        self.in_bar = False
        self.cur_rh = []
        self.cur_q = None
        self.hand = None
        self.pending_onset = None
        self.chromatic_count = 0
        self.phase = "header"
        self.header_buf = ""
        self.len_override = False
        self.bar_len = self.default_bar_len

    def _set_key(self, key):
        key = key.strip().strip(".,;")
        if not key or key == self.key:
            return
        try:
            self.scale = _scale(key)
        except Exception:
            return
        self.key = key

    def _set_ts(self, ts):
        self.default_bar_len = max(1, _ts_steps(ts, self.spb))
        if not self.len_override:
            self.bar_len = self.default_bar_len

    def _sync_fields(self):
        m = re.search(r"spb=\s*(\d+)", self.raw)
        if m:
            self.spb = int(m.group(1))
        keys = re.findall(r"key=\s*(\S+)", self.raw)
        if keys:
            self._set_key(keys[-1])
        else:
            m = re.search(r"Key:\s*([A-G](?:#|b)?\s*(?:major|minor))", self.raw, re.I)
            if m:
                self._set_key(m.group(1).replace(" ", "_"))
        if self.phase == "header":
            m = re.search(r"len=\s*(\d+)", self.header_buf)
            if m:
                self.len_override = True
                self.bar_len = max(1, int(m.group(1)))
                return
        where = self.header_buf if self.phase == "header" and "ts=" in self.header_buf else self.raw
        tss = re.findall(r"ts=\s*(\d+)\s*/\s*(\d+)", where)
        if not tss:
            tss = re.findall(r"Meter:\s*(\d+)\s*/\s*(\d+)", self.raw)
        if tss and not self.len_override:
            self._set_ts(f"{tss[-1][0]}/{tss[-1][1]}")

    def _copied_duration(self):
        """Duration that would make this right-hand bar identical to the previous one."""
        prev = self.prev_rh
        if self.hand != "<|rh|>" or self.pending_onset is None or not prev:
            return None
        if len(self.cur_rh) + 1 != len(prev):
            return None
        if tuple(self.cur_rh) != prev[:-1]:
            return None
        if self.pending_onset != prev[-1][0]:
            return None
        return prev[-1][1]

    def decisions(self):
        """Token ids to ban, and ids to subtract `pitch_penalty` from, for the next token."""
        bans, pens = [], []
        if self.prev_q:
            for bpm, i in self.tempo_ids.items():
                if bpm != self.prev_q and abs(bpm - self.prev_q) / self.prev_q <= self.tempo_ratio:
                    bans.append(i)
        if self.phase == "onset":
            for step, i in self.onset_ids.items():
                if step >= self.bar_len:
                    bans.append(i)
        if self.phase == "pitch" and self.scale:
            root, pcs = self.scale
            hard = self.chromatic_count >= self.chromatic_budget
            for pitch, i in self.pit_ids.items():
                if (pitch - root) % 12 not in pcs:
                    (bans if hard else pens).append(i)
            dur = self._copied_duration()
            if dur in self.dur_ids:
                bans.append(self.dur_ids[dur])
        return bans, pens

    def _apply(self, scores):
        n = scores.shape[-1]
        bans, pens = self.decisions()
        bans = [i for i in dict.fromkeys(bans) if 0 <= i < n]
        pens = [i for i in dict.fromkeys(pens) if 0 <= i < n and i not in bans]
        if bans:
            scores[:, bans] = float("-inf")
        if pens:
            scores[:, pens] = scores[:, pens] - self.pitch_penalty
        return scores
