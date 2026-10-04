# midi-composer

Fine-tune a small open LLM (with LoRA) to compose classical solo-piano music from
natural-language composition briefs.

Training data: 332 score-aligned piano MIDI files from piano-midi.de (Bernd Krueger,
CC BY-SA; check the license before redistributing data or models). They are not in
this repo.

## Run in Google Colab

Upload `notebooks/midi_composer_colab.ipynb` to Colab (File > Upload notebook), or
[open it from GitHub](https://colab.research.google.com/github/19katz/midi-composer/blob/main/notebooks/midi_composer_colab.ipynb).
It is self-contained: the `midicomposer` package is inlined as `%%writefile` cells.
`notebooks/pipeline.ipynb` is the same pipeline but clones this repo instead, which is
handier while developing. Regenerate the standalone copy after changing `src/` or
`pipeline.ipynb` with `python scripts/build_standalone.py` (a test checks it is current).

Pick a GPU runtime and follow the first cell's instructions:

- MIDI files go in Google Drive at `MyDrive/MIDI/`, or upload a zip when prompted.
- Add a Colab Secret `GEMINI_API_KEY` for real descriptions.
- Run once with `SMOKE_TEST = True`, then set it to `False` for a full run.

## Local setup

```sh
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest tests
.venv/bin/python scripts/roundtrip.py ~/MIDI --write-text data/text
```

## Layout

- `src/midicomposer/codec.py`: MIDI <-> text format (see the module docstring)
- `src/midicomposer/features.py`, `describe.py`: per-piece analysis and Gemini descriptions
- `src/midicomposer/dataset.py`: windowing, transposition, prompts, split by work
- `src/midicomposer/compose.py`: window-by-window generation
- `src/midicomposer/evaluate.py`: automatic checks on generated scores
- `scripts/roundtrip.py`: codec fidelity check over a MIDI directory
- `notebooks/pipeline.ipynb`: the end-to-end Colab pipeline

## Notes on the data

- The files never mark minor mode in their key signatures, so the codec infers major vs.
  relative minor from the final bass note, falling back to Krumhansl-Schmuckler.
- Codec fidelity over all 332 files: every note's pitch and grid onset survives
  encode -> decode, and 96% of files re-encode to identical text. The other 13 differ only
  in durations of notes re-struck while the same pitch is still held. Tempo is stored per
  bar, so rubato within a bar is dropped (median onset shift about 30 ms).
- Text costs about 1.25 characters per Qwen token: a median piece is ~14k tokens, so
  training uses windows of up to 32 bars (at most ~4.7k tokens each).
