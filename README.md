# midi-composer

Fine-tune a small open LLM (with LoRA) to compose classical solo-piano music from
natural-language composition briefs.

Training data: 332 score-aligned piano MIDI files from piano-midi.de (Bernd Krueger,
CC BY-SA; check the license before redistributing data or models). Expected at `~/MIDI`.

## Layout

- `src/midicomposer/codec.py`: MIDI <-> text format (see the module docstring)
- `scripts/roundtrip.py`: codec fidelity check over a MIDI directory
- `notebooks/pipeline.ipynb`: end-to-end pipeline (descriptions, dataset, LoRA, eval)

## Setup

```sh
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python scripts/roundtrip.py ~/MIDI --write-text data/text
```

## Codec status

Over all 332 files: every note's pitch and grid onset survives encode -> decode, and
96% of files re-encode to identical text. The other 13 files differ only in the
durations of notes re-struck while the same pitch is still held, which MIDI note-off
pairing cannot represent. Rubato within a bar is dropped (tempo is per bar), giving a
median onset shift of about 30 ms.
