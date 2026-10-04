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
- `src/midicomposer/runs.py`: test-set generation runs, saved outputs, and `results.jsonl`
- `scripts/roundtrip.py`: codec fidelity check over a MIDI directory
- `scripts/build_standalone.py`: generates the self-contained Colab notebook
- `notebooks/pipeline.ipynb`: the end-to-end pipeline (clones this repo)
- `notebooks/midi_composer_colab.ipynb`: the same pipeline with the library inlined

## How the data is tokenized

Tokenization happens in two stages: MIDI is converted to a text notation, and the base
model's own tokenizer then splits that text into tokens.

### Stage 1: MIDI to text (`codec.py`)

Each piece becomes a header followed by one bar line and two hand lines per bar:

```
<|piece_start|>
<|meta|> key=F_minor ts=12/8 bars=263 spb=12
<|plan|> Exposition:1-65 Development:66-135 ...
bar 37 @0.14 q126
R v6 0:C5:6 v5 6:Ab4.C5:6
L v4 0:F2.C3:36
...
<|piece_end|>
```

- **Header.** `<|meta|>` gives the key, the opening time signature, the total number of
  bars (so the model commits to a length up front), and the grid resolution. `<|plan|>`
  is the section map with bar ranges, taken from the description LLM; it is optional.
- **Bar lines.** The bar number, `@` its fractional position in the piece, and `q` the
  bar's average tempo in quarter notes per minute. `ts=` appears only when the time
  signature changes, and `len=` only when a bar's real length differs from what its time
  signature implies (cadenzas and other irregular bars).
- **Hand lines.** `R` and `L` come from the files' "Piano right" / "Piano left" tracks.
  Each note event is `onset:pitches:duration`, where the onset is measured from the start
  of the bar. Onset and duration are in grid steps of 1/12 of a quarter note, which covers
  sixteenths, triplets, and 32nd-note triplets. Notes with the same onset, duration, and
  velocity are grouped with `.` (`Ab4.C5`). `vN` sets the velocity level (0-7, from MIDI
  velocity // 16) for the events that follow it.
- **Why encode in ticks.** These files are score-aligned: 99.7% of onsets in the median
  file sit exactly on a 24-steps-per-beat grid in MIDI ticks, and the performer's timing is
  stored separately in a dense tempo map (a median of 662 tempo changes per piece). Encoding
  in ticks therefore recovers the written rhythm instead of approximating it from seconds.
- **What is lost.** Rubato within a bar (tempo is per bar; median onset shift ~30 ms),
  the sustain pedal (not yet encoded), and some durations of notes re-struck while the
  same pitch is still held, which MIDI note-off pairing cannot represent.

Before encoding, the main key is inferred, because these files never mark minor mode (see
below). `scripts/roundtrip.py` checks the codec over a whole MIDI directory.

### Stage 2: text to model tokens

The text goes through the base model's tokenizer (Qwen2.5) unchanged. No tokens are added
to the vocabulary. The tokenizer splits the notation finely, with every digit becoming its
own token:

```
"R v6 0:C5:4"  ->  ['R', 'Ġv', '6', 'Ġ', '0', ':C', '5', ':', '4']
```

This costs about 8.5 tokens per note, 130 per bar, and roughly 14k per median piece, with
the longest piece at about 130k. Whole pieces don't fit in a training sequence, so
`dataset.py` cuts them into windows:

- The first window's target is the header plus the opening bars. Each later window's
  prompt contains the brief, the header, and up to 8 previous bars (capped at ~1.2k
  tokens), and its target is up to the next 32 bars (capped at ~3k tokens).
- Every prompt starts with the composition brief, optionally with the key and meter
  (each is dropped 30% of the time so the model also works without them).
- Augmentation: each piece appears in its original key plus 2 random transpositions
  between -3 and +3 semitones. Shifts that push a note off the 88-key range are skipped,
  and the key in the header is updated to match.
- The longest training example is ~4.7k tokens, so `MAX_LEN` is 5120.

At generation time, `compose.py` uses the same prompts: it asks for the header and opening
bars, then for continuations, until the model writes `<|piece_end|>` or reaches the bar
count it planned.

Adding about 500 dedicated music tokens to the vocabulary (one each per pitch, onset,
duration, velocity, position, and tempo bin) would cut sequences to about 42% of their
current length, so the same window would hold ~75 bars instead of 32. That is the main
planned improvement. Each new token's embedding would start as the average of the
subword embeddings it replaces, and those rows would need training alongside LoRA.

## How LoRA is used

**LoRA (Low-Rank Adaptation)** fine-tunes a model without changing its original weights.
Each targeted weight matrix $W$ (size $d \times k$) stays frozen, and two small
matrices are trained instead: $B$ ($d \times r$) and $A$ ($r \times k$), with a
small rank $r$. The layer then computes

```
h = W x + (alpha / r) * B A x
```

so $BA$ acts as a learned low-rank correction to $W$ with far fewer parameters. Only
$A$ and $B$ receive gradients and optimizer state, which is what makes training fit
on one GPU.

Settings in the notebook's training cells:

| Setting | Value |
|---|---|
| Base model | `unsloth/Qwen2.5-3B-Instruct` (~3.1B parameters), via Unsloth |
| Quantization | Base weights loaded in 4-bit (QLoRA); adapters train in 16-bit |
| Rank, alpha | `r = 32`, `lora_alpha = 32` (scale 1), dropout 0 |
| Target layers | All linear layers in all 36 layers: `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention) and `gate_proj`, `up_proj`, `down_proj` (MLP) |
| Trainable parameters | ~60M, about 2% of the model; the saved adapter is ~120 MB |
| Loss | On the assistant response only (the score), via Unsloth's `train_on_responses_only`; the brief and context bars are masked |
| Optimizer, schedule | 8-bit AdamW, learning rate 2e-4, cosine decay with 3% warmup |
| Batch | 1 window per step, gradient accumulation 8 (effective batch 8) |
| Length, epochs | `MAX_LEN = 5120`, 1 epoch (30 steps in smoke-test mode) |
| Precision | bf16 where supported (A100, L4), otherwise fp16 (T4) |

Each training example is rendered with the Qwen chat template: a system message, the user
message (brief plus context), and the assistant message (the score window). Checkpoints are
written every 200 steps and training resumes from the latest one. The final adapter and
tokenizer are saved to `adapters/<run>/` in the working directory on Drive.

Why LoRA here: full fine-tuning of a 3B model needs optimizer state for every weight,
which doesn't fit on Colab GPUs. With only 332 pieces, freezing the base model also
limits overfitting and preserves its general language ability, which the briefs rely on.
Adapters are small, so several (for example, one per composer) can share one base model.

If the model underfits the notation, raise `r` to 64, or also train the embedding and
output layers. To scale up, change `BASE_MODEL` to Qwen2.5-7B on an A100. Moving to a
different model family also changes the token counts above and the chat-template markers
that `train_on_responses_only` looks for.

## Notes on the data

- The files never mark minor mode in their key signatures, so the codec infers major vs.
  relative minor from the final bass note, falling back to Krumhansl-Schmuckler.
- Codec fidelity over all 332 files: every note's pitch and grid onset survives
  encode -> decode, and 96% of files re-encode to identical text. The other 13 differ only
  in durations of notes re-struck while the same pitch is still held. Tempo is stored per
  bar, so rubato within a bar is dropped (median onset shift about 30 ms).
- Text costs about 1.25 characters per Qwen token: a median piece is ~14k tokens, so
  training uses windows of up to 32 bars (at most ~4.7k tokens each).
