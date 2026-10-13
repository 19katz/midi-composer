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
from midicomposer.vocab import approx_tokens, music_tokens, parse_pit, pit

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
    assert "<|ped_on|>" in text and "<|pit_C5|>" in text


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
    toks = music_tokens()
    assert len(toks) == len(set(toks))
    assert pit(60) in toks and pit(21) in toks and pit(108) in toks
    assert parse_pit(pit(61)) == 61
    assert approx_tokens("<|rh|> <|v_2|> <|t_0|> <|pit_C5|> <|d_6|>") == 5


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


def test_score_constraints_mask_key_rhythm_onset_and_tempo():
    import numpy as np
    from midicomposer.constraints import ScoreConstraints

    tok = _Tok()
    proc = ScoreConstraints(tok, pitch_penalty=8, chromatic_budget=2, tempo_ratio=0.08)
    # Prompt ends on a finished bar: RH eighth-eighth, tempo 100, 3/8 so the bar is 18 steps.
    # C# minor contains F#; C and D and F are the chromatic notes used below.
    opening = tok.ids(
        "Key", ":", " C#", " minor", "\n",
        "Meter", ":", " 3/8", "\n",
        "<|bar|>", "<|q_100|>",
        "<|rh|>", "<|t_0|>", "<|pit_Cs4|>", "<|d_6|>", "<|t_6|>", "<|pit_E4|>", "<|d_6|>",
        "<|lh|>",
    )
    scores = np.zeros((1, max(tok.i2t) + 1))
    proc(np.array([opening]), scores)
    assert proc.prev_rh == ((0, 6), (6, 6))
    assert proc.prev_q == 100
    assert proc.bar_len == 18

    # Continuation copies the first onset and duration, then the second onset.
    # The duration that would finish the copy is banned; the shared downbeat is not.
    copied = tok.ids("<|bar|>", "<|rh|>", "<|t_0|>", "<|pit_Cs4|>", "<|d_6|>", "<|t_6|>")
    proc(np.array([opening + copied]), scores)
    bans, pens = proc.decisions()
    assert tok.t2i["<|d_6|>"] in bans
    assert tok.t2i["<|t_0|>"] not in bans
    assert tok.t2i["<|q_100|>"] not in bans
    assert tok.t2i["<|q_101|>"] in bans  # 1% wobble
    assert tok.t2i["<|q_120|>"] not in bans  # real jump
    assert tok.t2i["<|t_18|>"] not in bans  # pitch position, not onset position
    fs = tok.t2i["<|pit_Fs4|>"]  # F# is outside C# minor (C# D# E F# G# A B) — F# IS in C# minor!
    # C# minor: C# D# E F# G# A B. Chromatic: C, D, F, G, A#...
    # Use C natural, pit_C4.
    cnat = tok.t2i["<|pit_C4|>"]
    assert cnat in pens and fs not in pens

    # Two chromatic notes fill the budget; the next out-of-key pitch is banned.
    more = tok.ids("<|pit_C4|>", "<|pit_D4|>")
    proc(np.array([opening + copied + more]), scores)
    bans, pens = proc.decisions()
    assert tok.t2i["<|pit_F4|>"] in bans
    assert fs not in bans

    # A different first duration breaks the copy, so the old completing duration is free.
    proc2 = ScoreConstraints(tok, pitch_penalty=8, chromatic_budget=2, tempo_ratio=0.08)
    proc2(np.array([opening]), np.zeros((1, max(tok.i2t) + 1)))
    diverged = tok.ids("<|bar|>", "<|rh|>", "<|t_0|>", "<|pit_Cs4|>", "<|d_8|>", "<|t_6|>")
    proc2(np.array([opening + diverged]), np.zeros((1, max(tok.i2t) + 1)))
    bans, _ = proc2.decisions()
    assert tok.t2i["<|d_6|>"] not in bans

    # Onset position: t_18 is past a 3/8 bar. len=10 shortens it further.
    proc3 = ScoreConstraints(tok)
    proc3(np.array([opening]), np.zeros((1, max(tok.i2t) + 1)))
    proc3(np.array([opening + tok.ids("<|bar|>", "<|rh|>")]), np.zeros((1, max(tok.i2t) + 1)))
    bans, _ = proc3.decisions()
    assert tok.t2i["<|t_18|>"] in bans and tok.t2i["<|t_0|>"] not in bans
    short = tok.ids("<|bar|>", "len", "=", "20")
    # "20" is the token; len=20 would be wrong because bar steps of 20 still bans t_20.
    # Feed characters that concatenate to len=10. The vocab has "20" only.
    # Use header pieces "len" "=" and a digit token. Add by consuming raw through known tokens:
    # token " " exists. Build len=10 from tokens we have: there is no "10" or "1" "0".
    # "20" makes len=20. Ban t_20, allow t_18.
    proc4 = ScoreConstraints(tok)
    proc4(np.array([opening]), np.zeros((1, max(tok.i2t) + 1)))
    proc4(np.array([opening + tok.ids("<|bar|>", "len", "=", "20", "<|rh|>")]), np.zeros((1, max(tok.i2t) + 1)))
    bans, _ = proc4.decisions()
    assert proc4.bar_len == 20
    assert tok.t2i["<|t_20|>"] in bans and tok.t2i["<|t_18|>"] not in bans
