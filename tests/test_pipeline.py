import glob
import io
import json
import os
import random

import pretty_midi
import pytest

from midicomposer.codec import decode, encode, transpose, transpose_key
from midicomposer.compose import compose
from midicomposer.dataset import build_examples, split_by_work, split_piece, windows, work_id
from midicomposer.describe import fallback_description
from midicomposer.evaluate import evaluate
from midicomposer.vocab import approx_tokens, degree_pitch, music_tokens, parse_pit, pitch_tokens, pit, tempo

MIDI_DIR = os.path.expanduser(os.environ.get("MIDI_DIR", "~/MIDI"))
FILES = sorted(glob.glob(os.path.join(MIDI_DIR, "*.mid")))
needs_midi = pytest.mark.skipif(not FILES, reason="no MIDI files")


@pytest.fixture(scope="module")
def piece():
    path = os.path.join(MIDI_DIR, "appass_1.mid")
    return path, encode(pretty_midi.PrettyMIDI(path))


def _note_keys(pm):
    from midicomposer.codec import SPB
    step = pm.resolution / SPB
    return {
        ("L" if "left" in inst.name.lower() else "R", round(pm.time_to_tick(n.start) / step), n.pitch)
        for inst in pm.instruments
        if not inst.is_drum and inst.name != "__dummy_sustain__"
        for n in inst.notes
    }


@needs_midi
def test_roundtrip_stable(piece):
    path, text = piece
    buf = io.BytesIO()
    decode(text).save(file=buf)
    buf.seek(0)
    pm2 = pretty_midi.PrettyMIDI(buf)
    assert _note_keys(pretty_midi.PrettyMIDI(path)) == _note_keys(pm2)
    text2 = encode(pm2)
    assert text.count("<|ped|>") == text2.count("<|ped|>")
    assert text2.splitlines()[0] == "<|piece_start|>" and "<|rh|>" in text2


def test_synthetic_roundtrip_is_byte_identical():
    mid = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(0, name="Piano right")
    inst.notes.append(pretty_midi.Note(velocity=80, pitch=72, start=0.0, end=0.5))
    mid.instruments.append(inst)
    inst = pretty_midi.Instrument(0, name="Piano left")
    inst.notes.append(pretty_midi.Note(velocity=64, pitch=48, start=0.0, end=1.0))
    inst.control_changes.append(pretty_midi.ControlChange(number=64, time=0.0, value=127))
    inst.control_changes.append(pretty_midi.ControlChange(number=64, time=0.9, value=0))
    mid.instruments.append(inst)
    text = encode(mid)
    buf = io.BytesIO()
    decode(text).save(file=buf)
    buf.seek(0)
    assert encode(pretty_midi.PrettyMIDI(buf)) == text
    assert "<|ped_on|>" in text and "<|sd_1|>" in text and "<|pit_" not in text


@needs_midi
def test_transpose_inverts(piece):
    _, text = piece
    up = transpose(text, 2)
    assert up is not None and up != text
    assert transpose(up, -2) == text
    assert transpose(text, 60) is None


def test_standalone_notebook_is_current():
    import subprocess
    import sys

    script = os.path.join(os.path.dirname(__file__), "..", "scripts", "build_standalone.py")
    r = subprocess.run([sys.executable, script, "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_surface_metrics_key_rhythm_silence_tempo():
    from midicomposer.evaluate import surface_metrics

    text = """<|piece_start|>
<|meta|> key=C_Major ts=4/4 bars=3 spb=12
<|bar|> 1 <|pos_0|> <|q_100|> ts=4/4
<|rh|> <|v_4|> <|t_0|> <|pit_C4|> <|d_48|>
<|lh|>
<|bar|> 2 <|pos_33|> <|q_100|>
<|rh|> <|v_4|> <|t_0|> <|pit_C4|> <|d_48|>
<|lh|>
<|bar|> 3 <|pos_66|> <|q_140|>
<|rh|> <|v_4|> <|t_0|> <|pit_Fs4|> <|d_24|>
<|lh|>
<|piece_end|>
"""
    m = surface_metrics(text)
    assert m["rhythm_repeat_frac"] == 0.5  # bar 2 copies bar 1; bar 3 does not
    assert m["silence_frac"] == 0.167
    assert m["tempo_jump_frac"] == 0.5  # 100 -> 140 is a jump; 100 -> 100 is not
    assert 0.6 < m["diatonic_frac"] < 0.9  # F# is the only chromatic note


def test_music_vocab_is_unique_and_covers_piano():
    from midicomposer.vocab import tempo_centers

    toks = music_tokens()
    assert len(toks) == len(set(toks))
    assert "<|sd_1|>" in toks and "<|sd_7|>" in toks and "<|oct_4|>" in toks
    assert "<|tempo_hold|>" in toks and "<|mel|>" in toks and "<|ch_V|>" in toks
    assert pit(60) not in toks
    assert parse_pit(pit(61)) == 61
    # 20 BPM to 260 BPM in 8% steps is a few dozen bins, not one token per BPM.
    assert 30 <= len(tempo_centers()) <= 40
    assert all(tempo(c) in toks for c in tempo_centers())
    assert approx_tokens("<|rh|> <|v_2|> <|t_0|> <|sd_1|> <|oct_5|> <|d_6|>") == 6


def test_canonical_pitch_roundtrip():
    for key in ("C_Major", "A_minor", "F#_Major", "Bb_minor"):
        for midi in range(21, 109):
            toks = pitch_tokens(key, midi)
            acc = sd = octave = 0
            for tok in toks:
                if tok.startswith("<|acc_"):
                    acc = int(tok[6:-2])
                elif tok.startswith("<|sd_"):
                    sd = int(tok[5:-2])
                elif tok.startswith("<|oct_"):
                    octave = int(tok[6:-2])
            assert degree_pitch(key, sd, octave, acc) == midi
            assert sum(tok.startswith("<|acc_") for tok in toks) <= 1


def test_transpose_key():
    assert transpose_key("F_minor", 2) == "G_minor"
    assert transpose_key("B_Major", 1) == "C_Major"


@needs_midi
def test_windows_cover_every_bar(piece):
    _, text = piece
    _, bars = split_piece(text)
    ws = list(windows(text, [{"name": "Exposition", "bars": [1, 65]}]))
    got = []
    for header, ctx, target, next_bar in ws:
        assert (header is None) == (next_bar == 1)
        got += split_piece("\n".join(["<|meta|>", target]))[1]
    assert got == bars
    assert ws[0][2].startswith("<|piece_start|>\n<|meta|>") and "<|plan|> Exposition:1-65" in ws[0][2]
    assert ws[-1][2].endswith("<|piece_end|>")


@needs_midi
def test_build_examples(piece):
    path, text = piece
    desc = fallback_description(path)
    ex = build_examples(text, desc, semitones=(-1, 0, 1), rng=random.Random(1))
    assert {e["semis"] for e in ex} == {-1, 0, 1}
    assert all(e["prompt"][-1]["role"] == "user" and e["completion"][0]["role"] == "assistant" for e in ex)


def test_work_split_keeps_movements_together():
    files = ["appass_1.mid", "appass_2.mid", "appass_3.mid", "elise.mid", "mz_570_1.mid", "mz_570_2.mid"] * 1
    assert work_id("mz_570_3.mid") == "mz_570"
    train, test = split_by_work(files, test_frac=0.34)
    assert not {work_id(f) for f in train} & {work_id(f) for f in test}


def test_rerank_treats_a_missing_rate_as_a_miss():
    from midicomposer.runs import _rerank_distance

    assert _rerank_distance({"diatonic_frac": 0.85, "rhythm_repeat_frac": None, "silence_frac": 0.05, "tempo_jump_frac": 0.1}) == 1.0
    assert _rerank_distance({}) == 4.0


def test_archive_stale_results(tmp_path):
    from midicomposer.runs import archive_stale_results, load_results

    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps({"tag": "old", "metrics": {"bars": 1}}) + "\n")
    dest = archive_stale_results(str(path))
    assert dest and dest.endswith(".jsonl") and not path.exists()
    assert load_results(dest)[0]["tag"] == "old"
    path.write_text(json.dumps({"tag": "new", "metrics": {"notes_per_bar": 3}}) + "\n")
    assert archive_stale_results(str(path)) is None


@needs_midi
def test_run_eval_saves_logs_and_resumes(piece, tmp_path):
    from midicomposer.runs import load_results, run_eval

    path, text = piece
    name = os.path.basename(path)
    targets = [w[2] for w in windows(text)]
    state = {}

    def replay(messages):
        if "Write the score header" in messages[-1]["content"]:
            state["it"] = iter(targets)
        return next(state["it"])

    args = dict(files=[name], texts={name: text}, descs={name: fallback_description(path)},
                out_dir=str(tmp_path), tag="ckpt-a", seeds=(0, 1), render=False)
    rows = run_eval(replay, **args)
    assert [r["seed"] for r in rows] == [0, 1]
    for r in rows:
        assert os.path.exists(r["paths"]["mid"]) and os.path.exists(r["paths"]["txt"])
        assert r["metrics"]["valid_token_frac"] == 1.0 and r["brief"]
    assert run_eval(replay, **args) == []
    assert len(load_results(str(tmp_path / "results.jsonl"))) == 2
    assert len(run_eval(replay, **{**args, "tag": "ckpt-b", "seeds": (0,)})) == 1


def test_piece_stop_ends_on_the_bar_after_the_budget():
    from midicomposer.compose import PieceStop

    class _Seq:
        def __init__(self, ids):
            self.ids = list(ids)
            self.shape = (len(self.ids),)

        def __getitem__(self, item):
            return _Seq(self.ids[item])

        def tolist(self):
            return list(self.ids)

    class _Batch:
        def __init__(self, ids):
            self.row = _Seq(ids)

        def __getitem__(self, _i):
            return self.row

    # Prompt is 3 tokens. max_bars=2 stops when the third bar token arrives.
    stop = PieceStop(prompt_len=3, bar_id=1, end_id=9, max_bars=2)
    assert stop(_Batch([0, 0, 0, 1, 2]), None) is False
    assert stop(_Batch([0, 0, 0, 1, 2, 1, 2]), None) is False
    assert stop(_Batch([0, 0, 0, 1, 2, 1, 2, 1]), None) is True

    done = PieceStop(prompt_len=1, bar_id=1, end_id=9, max_bars=8)
    assert done(_Batch([0, 9]), None) is True


@needs_midi
def test_compose_with_replayed_model(piece):
    """A fake model that replays the real piece window by window must reassemble it."""
    path, text = piece
    targets = [w[2] for w in windows(text)]
    calls = iter(targets)
    out = compose(lambda messages: next(calls), "Write a sonata movement.")
    assert split_piece(out)[1] == split_piece(text)[1]
    m = evaluate(out, requested_key="F Minor", reference_text=text)
    assert m["valid_token_frac"] == 1.0 and m["key_match"] and m["pc_hist_l1_to_reference"] < 1e-9


class _Tok:
    def __init__(self):
        from midicomposer.vocab import music_tokens
        tokens = music_tokens() + [
            "Key", ":", " C#", " minor", "\n", "Meter", " 3/8", " ",
            "len", "=", "20", "ts", "4", "/", "4",
        ]
        self.i2t = {i + 1: t for i, t in enumerate(tokens)}
        self.t2i = {t: i for i, t in self.i2t.items()}
        self.unk_token_id = 0

    def convert_tokens_to_ids(self, tok):
        return self.t2i.get(tok, 0)

    def convert_ids_to_tokens(self, i):
        return self.i2t.get(int(i), "")

    def ids(self, *tokens):
        return [self.t2i[t] for t in tokens]


def _penalized(groups, amount=None):
    ids = []
    for penalty, group in groups:
        if amount is None or penalty == amount:
            ids.extend(group)
    return ids


def test_score_constraints_mask_key_rhythm_onset_and_tempo():
    import numpy as np
    from midicomposer.constraints import ScoreConstraints
    from midicomposer.vocab import tempo_centers

    tok = _Tok()
    qtok = tempo(100)
    qval = int(qtok[4:-2])
    proc = ScoreConstraints(tok, pitch_penalty=2, chromatic_budget=2, rhythm_penalty=4, tempo_ratio=0.08)
    # Prompt ends on a finished bar. C# minor, RH eighth-eighth, 3/8 so the bar is 18 steps.
    opening = tok.ids(
        "Key", ":", " C#", " minor", "\n",
        "Meter", ":", " 3/8", "\n",
        "<|bar|>", qtok,
        "<|rh|>", "<|t_0|>", "<|sd_1|>", "<|oct_4|>", "<|d_6|>",
        "<|t_6|>", "<|sd_3|>", "<|oct_4|>", "<|d_6|>",
        "<|lh|>",
    )
    scores = np.zeros((1, max(tok.i2t) + 1))
    proc(np.array([opening]), scores)
    assert proc.prev_events["<|rh|>"] == ((0, 6), (6, 6))
    assert proc.prev_q == qval
    assert proc.bar_len == 18

    # Same eighth-note beat as the previous bar is penalized, including the downbeat.
    # It is not banned: real pieces repeat a beat about a fifth of the time.
    beat = ScoreConstraints(tok, pitch_penalty=2, chromatic_budget=2, rhythm_penalty=4, tempo_ratio=0.08)
    beat(np.array([opening]), scores)
    beat(np.array([opening + tok.ids("<|bar|>", "<|rh|>")]), scores)
    bans, groups = beat.decisions()
    assert tok.t2i["<|t_0|>"] in _penalized(groups, 4)
    assert tok.t2i["<|t_0|>"] not in bans
    assert tok.t2i["<|t_6|>"] not in _penalized(groups, 4)

    # The duration that would finish an exact copy is penalized the same way.
    copied = tok.ids("<|bar|>", "<|rh|>", "<|t_0|>", "<|sd_1|>", "<|oct_4|>", "<|d_6|>", "<|t_6|>")
    proc(np.array([opening + copied]), scores)
    bans, groups = proc.decisions()
    assert tok.t2i["<|d_6|>"] in _penalized(groups, 4)
    assert tok.t2i["<|d_6|>"] not in bans
    assert tok.t2i[qtok] not in bans
    near = [c for c in tempo_centers() if c != qval and abs(c - qval) / qval <= 0.08]
    far = [c for c in tempo_centers() if abs(c - qval) / qval > 0.08]
    assert near and all(tok.t2i[f"<|q_{c}|>"] in bans for c in near)
    assert any(tok.t2i[f"<|q_{c}|>"] not in bans for c in far)
    # Accidentals are penalized, not removed. Scale degrees of the key are not.
    assert tok.t2i["<|acc_-1|>"] in _penalized(groups, 2)
    assert tok.t2i["<|sd_1|>"] not in _penalized(groups, 2)

    # Two accidentals fill the budget; another accidental is then banned.
    more = tok.ids("<|acc_-1|>", "<|acc_1|>")
    proc(np.array([opening + copied + more]), scores)
    bans, groups = proc.decisions()
    assert tok.t2i["<|acc_2|>"] in bans
    assert tok.t2i["<|sd_1|>"] not in bans

    # A different first duration is not an exact copy, so the old duration is free.
    proc2 = ScoreConstraints(tok, pitch_penalty=2, chromatic_budget=2, rhythm_penalty=4, tempo_ratio=0.08)
    proc2(np.array([opening]), np.zeros((1, max(tok.i2t) + 1)))
    diverged = tok.ids("<|bar|>", "<|rh|>", "<|t_0|>", "<|sd_1|>", "<|oct_4|>", "<|d_8|>", "<|t_6|>")
    proc2(np.array([opening + diverged]), np.zeros((1, max(tok.i2t) + 1)))
    _bans, groups = proc2.decisions()
    assert tok.t2i["<|d_6|>"] not in _penalized(groups, 4)

    # Onset at or past the bar is still a hard ban. len=20 shortens a longer bar.
    proc3 = ScoreConstraints(tok)
    proc3(np.array([opening]), np.zeros((1, max(tok.i2t) + 1)))
    proc3(np.array([opening + tok.ids("<|bar|>", "<|rh|>")]), np.zeros((1, max(tok.i2t) + 1)))
    bans, _groups = proc3.decisions()
    assert tok.t2i["<|t_18|>"] in bans and tok.t2i["<|t_0|>"] not in bans
    proc4 = ScoreConstraints(tok)
    proc4(np.array([opening]), np.zeros((1, max(tok.i2t) + 1)))
    proc4(np.array([opening + tok.ids("<|bar|>", "len", "=", "20", "<|rh|>")]), np.zeros((1, max(tok.i2t) + 1)))
    bans, _groups = proc4.decisions()
    assert proc4.bar_len == 20
    assert tok.t2i["<|t_20|>"] in bans and tok.t2i["<|t_18|>"] not in bans

    # A chord token penalizes degrees outside that chord. I in C# minor is C# E G#.
    proc5 = ScoreConstraints(tok, pitch_penalty=2, chromatic_budget=16)
    proc5(np.array([opening]), np.zeros((1, max(tok.i2t) + 1)))
    proc5(np.array([opening + tok.ids("<|bar|>", "<|ch_I|>", "<|rh|>", "<|t_0|>")]), np.zeros((1, max(tok.i2t) + 1)))
    _bans, groups = proc5.decisions()
    pitched = _penalized(groups, 2)
    assert tok.t2i["<|sd_2|>"] in pitched and tok.t2i["<|sd_1|>"] not in pitched


def test_weighted_loss_upweights_melody_not_figuration():
    torch = pytest.importorskip("torch")
    from midicomposer.weighted_loss import _ids, _weights

    tok = _Tok()
    special = _ids(tok, lambda t: (
        t.startswith("<|ch_") or t.startswith("<|q_")
        or t in ("<|rhythm_same|>", "<|rhythm_new|>", "<|tempo_hold|>", "<|phrase|>", "<|mel|>")
    ))
    boundary = _ids(tok, lambda t: t in ("<|rh|>", "<|lh|>", "<|ped|>", "<|bar|>", "<|piece_end|>"))
    mel = tok.t2i["<|mel|>"]
    sd = tok.t2i["<|sd_1|>"]
    labels = torch.tensor([[mel, sd, tok.t2i["<|rh|>"], sd, -100]])
    w = _weights(labels, special, boundary, mel, 3.0)
    assert w[0].tolist() == [3.0, 3.0, 1.0, 1.0, 0.0]


def test_new_token_rows_receive_gradient(tmp_path):
    torch = pytest.importorskip("torch")
    from torch import nn
    from midicomposer.token_rows import attach_new_token_training, load_new_token_rows, save_new_token_rows

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(6, 3)
            self.head = nn.Linear(3, 6, bias=False)

        def get_input_embeddings(self):
            return self.embed

        def get_output_embeddings(self):
            return self.head

        def forward(self, ids):
            return self.head(self.embed(ids))

    model = Tiny()
    attach_new_token_training(model, 4, 2)
    assert model.embed.weight.requires_grad is False
    assert model.embed.new_token_rows.requires_grad
    model(torch.tensor([[4, 5]])).sum().backward()
    assert model.embed.new_token_rows.grad.abs().sum() > 0
    assert model.head.new_token_rows.grad.abs().sum() > 0
    path = tmp_path / "new_token_rows.pt"
    save_new_token_rows(model, path)
    fresh = Tiny()
    load_new_token_rows(fresh, path)
    assert torch.allclose(fresh.embed.new_token_rows, model.embed.new_token_rows)
