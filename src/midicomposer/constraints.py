"""Soft scores for a model that already writes valid score text.

Applied inside `model.generate`. Real pieces sit in the middle of these axes
(about 83% diatonic, 19% repeated rhythm, 8% silence), so the defaults penalize
the extremes instead of banning them:

- accidentals and non-chord degrees get `pitch_penalty`. A hard ban only
  starts after `chromatic_budget` accidentals in the bar.
- an onset whose eighth-note beat matches the previous bar, at the same place in
  the bar, gets `rhythm_penalty`. The duration that would finish an exact copy does too.
  Both amounts are subtracted from the logits and then divided by the sampling
  temperature, so 2 and 4 behave like bans (about 9x and 85x at temperature 0.9).
  The defaults are a nudge: chromatic notes and a repeated beat stay possible.
- ending a bar that is still mostly empty gets `silence_penalty`. An empty bar is left alone.
- an onset at or past the bar length is still banned. So is a tempo token within
  `tempo_ratio` of the previous bar, other than the previous tempo itself.
"""

import re

from .codec import _ts_steps
from .evaluate import _scale
from .vocab import chord_pcs, music_tokens, parse_pit, parse_tagged_int, scale_pcs

try:
    from transformers import LogitsProcessor
except ImportError:  # CPU tests, and any environment without the training stack
    class LogitsProcessor:
        pass


_MUSIC = re.compile(r"<\|[^|]+\|>")
_HANDS = ("<|mel|>", "<|rh|>", "<|lh|>")


class ScoreConstraints(LogitsProcessor):
    def __init__(self, tokenizer, pitch_penalty=0.5, chromatic_budget=16, rhythm_penalty=1.0,
                 silence_penalty=2.0, silence_limit=0.15, tempo_ratio=0.08):
        self.tokenizer = tokenizer
        self.pitch_penalty = float(pitch_penalty)
        self.chromatic_budget = int(chromatic_budget)
        self.rhythm_penalty = float(rhythm_penalty)
        self.silence_penalty = float(silence_penalty)
        self.silence_limit = float(silence_limit)
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
        self.acc_ids = [i for t, i in self.ids.items() if t.startswith("<|acc_")]
        self.sd_ids = {parse_tagged_int(t, "sd"): i for t, i in self.ids.items() if t.startswith("<|sd_")}
        self.bar_id = self.ids.get("<|bar|>")
        self.end_id = self.ids.get("<|piece_end|>")
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
        self.prev_events = {}
        self.cur_events = {h: [] for h in _HANDS}
        self.pending_onset = None
        self.chromatic_count = 0
        self.chord = None
        self.covered = set()
        self.note_count = 0

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
        elif tok.startswith("<|ch_"):
            name = tok[5:-2]
            self.chord = chord_pcs(self.key, name) if self.key else None
        elif tok in ("<|mel|>", "<|rh|>", "<|lh|>", "<|ped|>"):
            self.hand = tok
            self.phase = "onset"
            self.pending_onset = None
        elif tok.startswith("<|v_") or tok in ("<|ped_on|>", "<|ped_off|>"):
            self.phase = "onset"
        elif tok.startswith("<|t_"):
            self.pending_onset = parse_tagged_int(tok, "t")
            if self.hand in _HANDS:
                self.phase = "pitch"
        elif tok.startswith("<|acc_"):
            self.chromatic_count += 1
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
            if self.hand in _HANDS and self.pending_onset is not None:
                self.cur_events.setdefault(self.hand, []).append((self.pending_onset, dur))
                if self.bar_len:
                    self.covered.update(range(max(0, self.pending_onset), min(self.bar_len, self.pending_onset + dur)))
                self.note_count += 1
            self.pending_onset = None
            self.phase = "onset"
        elif tok.startswith("<|q_"):
            self.cur_q = parse_tagged_int(tok, "q")

    def _commit_bar(self):
        if any(self.cur_events.get(h) for h in _HANDS):
            self.prev_events = {h: tuple(self.cur_events.get(h, ())) for h in _HANDS}
        if self.cur_q:
            self.prev_q = self.cur_q

    def _on_bar(self):
        if self.in_bar:
            self._commit_bar()
        self.cur_events = {h: [] for h in _HANDS}
        self.cur_q = None
        self.hand = None
        self.pending_onset = None
        self.chromatic_count = 0
        self.chord = None
        self.covered = set()
        self.note_count = 0
        self.header_buf = ""
        self.len_override = False
        self.bar_len = self.default_bar_len
        self.in_bar = True
        self.phase = "header"

    def _finish_prompt(self):
        """Context bars in the prompt are complete. The next token starts the continuation."""
        if self.in_bar:
            self._commit_bar()
        self.in_bar = False
        self.cur_events = {h: [] for h in _HANDS}
        self.cur_q = None
        self.hand = None
        self.pending_onset = None
        self.chromatic_count = 0
        self.chord = None
        self.covered = set()
        self.note_count = 0
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
        """Duration that would make this hand an exact copy of the previous bar."""
        prev = self.prev_events.get(self.hand) or ()
        cur = self.cur_events.get(self.hand) or []
        if self.pending_onset is None or not prev or len(cur) + 1 != len(prev):
            return None
        if tuple(cur) != prev[:-1] or self.pending_onset != prev[-1][0]:
            return None
        return prev[-1][1]

    def decisions(self):
        """Hard bans, then `(penalty, token ids)` groups, for the next token."""
        bans, groups = [], []

        def penalize(amount, ids):
            ids = [i for i in ids if isinstance(i, int)]
            if ids:
                groups.append((float(amount), ids))

        if self.prev_q:
            for bpm, i in self.tempo_ids.items():
                if bpm != self.prev_q and abs(bpm - self.prev_q) / self.prev_q <= self.tempo_ratio:
                    bans.append(i)
        if self.phase == "onset":
            for step, i in self.onset_ids.items():
                if step >= self.bar_len:
                    bans.append(i)
            prev = self.prev_events.get(self.hand) or ()
            cur = self.cur_events.get(self.hand) or []
            if len(cur) < len(prev):
                grid = max(1, self.spb // 2)
                beat = prev[len(cur)][0] // grid
                penalize(self.rhythm_penalty, [i for step, i in self.onset_ids.items() if step // grid == beat])
            if self.note_count and self.bar_len and 1 - len(self.covered) / self.bar_len > self.silence_limit:
                penalize(self.silence_penalty, [self.bar_id, self.end_id])
        if self.phase == "pitch":
            hard = self.chromatic_count >= self.chromatic_budget
            if hard:
                bans.extend(self.acc_ids)
            else:
                penalize(self.pitch_penalty, self.acc_ids)
            if self.chord and self.key:
                root, intervals = scale_pcs(self.key)
                outside = []
                for sd, i in self.sd_ids.items():
                    if (root + intervals[sd - 1]) % 12 not in self.chord:
                        outside.append(i)
                penalize(self.pitch_penalty, outside)
            if self.scale:
                root, pcs = self.scale
                off = [i for pitch, i in self.pit_ids.items() if (pitch - root) % 12 not in pcs]
                if hard:
                    bans.extend(off)
                else:
                    penalize(self.pitch_penalty, off)
            dur = self._copied_duration()
            if dur in self.dur_ids:
                penalize(self.rhythm_penalty, [self.dur_ids[dur]])
        return bans, groups

    def _apply(self, scores):
        n = scores.shape[-1]
        bans, groups = self.decisions()
        banned = {i for i in bans if 0 <= i < n}
        applied = []
        for amount, ids in groups:
            keep = []
            for i in dict.fromkeys(ids):
                if 0 <= i < n and i not in banned:
                    keep.append(i)
                    banned.add(i)  # a token is penalized once, by the first group that names it
            if keep:
                applied.append((amount, keep))
        banned = [i for i in dict.fromkeys(bans) if 0 <= i < n]
        # A Python-list index builds a CPU index tensor. Unsloth's sampler is a
        # Triton kernel and rejects that with "cpu tensor?".
        index_fill = getattr(scores, "index_fill_", None)
        if index_fill is None:
            if banned:
                scores[:, banned] = float("-inf")
            for amount, ids in applied:
                scores[:, ids] = scores[:, ids] - amount
            return scores
        import torch
        device, dtype = scores.device, scores.dtype
        if banned:
            scores.index_fill_(1, torch.tensor(banned, device=device, dtype=torch.long), torch.finfo(dtype).min)
        for amount, ids in applied:
            idx = torch.tensor(ids, device=device, dtype=torch.long)
            delta = torch.full((scores.shape[0], idx.numel()), -amount, device=device, dtype=dtype)
            scores.index_add_(1, idx, delta)
        return scores
