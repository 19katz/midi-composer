import glob
import io
import os
import random

import pretty_midi
import pytest

from midicomposer.codec import decode, encode, transpose, transpose_key
from midicomposer.compose import compose
from midicomposer.dataset import build_examples, split_by_work, split_piece, windows, work_id
from midicomposer.describe import fallback_description
from midicomposer.evaluate import evaluate

MIDI_DIR = os.path.expanduser(os.environ.get("MIDI_DIR", "~/MIDI"))
FILES = sorted(glob.glob(os.path.join(MIDI_DIR, "*.mid")))
needs_midi = pytest.mark.skipif(not FILES, reason="no MIDI files")


@pytest.fixture(scope="module")
def piece():
    path = os.path.join(MIDI_DIR, "appass_1.mid")
    return path, encode(pretty_midi.PrettyMIDI(path))


@needs_midi
def test_roundtrip_stable(piece):
    _, text = piece
    buf = io.BytesIO()
    decode(text).save(file=buf)
    buf.seek(0)
    assert encode(pretty_midi.PrettyMIDI(buf)) == text


@needs_midi
def test_transpose_inverts(piece):
    _, text = piece
    up = transpose(text, 2)
    assert up is not None and up != text
    assert transpose(up, -2) == text
    assert transpose(text, 60) is None


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
