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

- `src/midicomposer/vocab.py`: dedicated music tokens added to the base LLM vocabulary
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

Two stages: MIDI becomes a text score, then that text is tokenized with Qwen's vocabulary
**plus about 900 music tokens** from `vocab.py`.

### Stage 1: MIDI to text (`codec.py`)

```
<|piece_start|>
<|meta|> key=A_minor ts=3/8 bars=126 spb=12
<|bar|> 2 <|pos_1|> <|q_69|>
<|rh|> <|v_2|> <|t_0|> <|pit_E5|> <|d_3|> <|t_3|> <|pit_Ds5|> <|d_3|> <|t_6|> <|pit_E5|> <|d_3|>
<|lh|>
<|ped|> <|t_0|> <|ped_on|> <|t_12|> <|ped_off|>
<|piece_end|>
```

- **Header.** Key, opening meter, total bars, grid resolution (`spb=12` steps per quarter
  note). Optional `<|plan|>` holds section names and bar ranges from the description LLM.
- **Bar line.** The bar number stays ordinary digits (the base model already counts).
  `<|pos_N|>` is N% through the piece; `<|q_N|>` is that bar's average tempo.
  `ts=` / `len=` appear only when meter changes or a bar is irregular (cadenzas).
- **Hands.** `<|rh|>` / `<|lh|>` from the files' right/left tracks. Each event is
  `<|t_onset|>` one or more `<|pit_...|>` then `<|d_duration|>`. Chord tones share
  onset and duration. `<|v_0|>`–`<|v_7|>` set loudness until the next velocity token.
  Sharps are spelled `s` (`C#4` → `<|pit_Cs4|>`) so each of the 88 piano keys is one token.
- **Pedal.** Optional `<|ped|>` line: sustain CC64, binarized to on/off, on the same grid.
- **What is lost.** Rubato within a bar (~30 ms median onset shift), continuous pedal
  values (only down/up), and some durations of notes re-struck while the same pitch is
  still held.

### Stage 2: music tokens on top of Qwen

`music_tokens()` is a fixed list of about 900 strings. Unsloth's `add_new_tokens(...,
method="interpolation")` adds them before LoRA and initializes each embedding as a blend
of the mean vocabulary vector and the mean of the subword pieces it replaces (`<|pit_C5|>`
starts near `:C` + `5`). The English brief still uses Qwen's existing tokens.

| Family | Count | Example |
|---|---|---|
| Structure | 10 | `<|bar|>`, `<|rh|>`, `<|piece_end|>` |
| Velocity | 8 | `<|v_0|>` … `<|v_7|>` |
| Onset | 193 | `<|t_0|>` … `<|t_192|>` |
| Duration | 256 | `<|d_1|>` … `<|d_256|>` |
| Tempo | 241 | `<|q_20|>` … `<|q_260|>` |
| Position | 101 | `<|pos_0|>` … `<|pos_100|>` |
| Pitch | 88 | `<|pit_A0|>` … `<|pit_C8|>` |

The `<|...|>` spelling cannot appear inside a brief, so a token like `<|d_6|>` will not
match the word "add6". Bar numbers, key names, and meters stay as ordinary text.

A note is now about 4 tokens (`t`, `pit`, `d`, plus velocity when it changes) instead of
~8.5. Training windows target ~3k music tokens and up to 64 bars, with up to 16 bars of
context. `MAX_LEN` is still 5120. At generation time `compose.py` uses the same prompts
and continues until `<|piece_end|>` or the planned bar count.

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
| Target layers | Attention and MLP (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) plus `embed_tokens` and `lm_head` so the new music tokens can be learned |
| Trainable parameters | ~70M LoRA weights; the saved adapter also stores the resized tokenizer |
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

If the model underfits the notation, raise `r` to 64. To scale up, change `BASE_MODEL`
to Qwen2.5-7B on an A100. A different model family also changes the chat-template markers
that `train_on_responses_only` looks for; `music_tokens()` stays the same.
When loading a saved adapter, call `add_new_tokens` with `music_tokens()` on the base
model **before** loading the adapter, using the same token list and order.

## Notes on the data

- The files never mark minor mode in their key signatures, so the codec infers major vs.
  relative minor from the final bass note, falling back to Krumhansl-Schmuckler.
- Codec fidelity over all 332 files: every note's pitch and grid onset survives
  encode -> decode, and 96% of files re-encode to identical text. The other 13 differ only
  in durations of notes re-struck while the same pitch is still held. Tempo is stored per
  bar, so rubato within a bar is dropped (median onset shift about 30 ms).
- Encoded scores are cached under `text_tok1/` so older colon-separated encodings are not
  reused after the vocabulary change.
