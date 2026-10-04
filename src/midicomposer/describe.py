"""Composition-brief descriptions of each piece from an audio-capable LLM (Gemini)."""

import json
import os
import shutil
import subprocess
import time

from .features import extract_features

DEFAULT_SOUNDFONT = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
COMPOSERS = {
    "albeniz": "Albéniz", "bach": "Bach", "balakire": "Balakirev", "beethoven": "Beethoven",
    "borodin": "Borodin", "brahms": "Brahms", "burgm": "Burgmüller", "chopin": "Chopin",
    "clementi": "Clementi", "debussy": "Debussy", "godowsky": "Godowsky", "granados": "Granados",
    "grieg": "Grieg", "haydn": "Haydn", "liszt": "Liszt", "mendelssohn": "Mendelssohn",
    "moszkowski": "Moszkowski", "mozart": "Mozart", "mussorgsk": "Mussorgsky", "rachmanin": "Rachmaninoff",
    "ravel": "Ravel", "schubert": "Schubert", "schumann": "Schumann", "sinding": "Sinding",
    "tschaikowsky": "Tchaikovsky", "tchaikovsky": "Tchaikovsky",
}
TEMPO_WORDS = (
    "adagio", "allegr", "andant", "grave", "larghetto", "largo", "lento", "maestoso", "moderato",
    "presto", "vivace", "vivo", "tempo", "agitato", "con moto", "langsam", "lebhaft", "rasch", "mässig",
)
VARIANTS = ("brief", "paragraph", "detailed", "styled_brief", "styled_paragraph", "styled_detailed")

PROMPT = """You are writing a commission brief that asks a composer to write this piece of
solo piano music. You are given a recording, the score's embedded metadata, and per-bar
analysis (note count, mean MIDI velocity, lowest/highest MIDI pitch).

Return JSON with exactly these keys:
- "brief": one imperative sentence ("Write a ...").
- "paragraph": 3-5 sentences on character, mood, tempo, meter, form, texture, and difficulty.
- "detailed": a section-by-section brief of the emotional and technical progression, giving
  approximate positions in the piece ("around 40% of the way through ..."): harmonic language,
  melodic character, figuration, texture, dynamics, climaxes, and the ending.
- "styled_brief", "styled_paragraph", "styled_detailed": the same three, but also naming the
  composer and the style or genre to emulate ("in the manner of Chopin's nocturnes").
- "sections": a list of {"name": str, "bars": [first, last]} covering bars 1..n_bars in order,
  using musical names (Introduction, Exposition, Theme, Variation 1, Trio, Coda, ...).

Rules: never name the specific piece, its title, or its catalog number. Do not name the key
or the exact tempo in the prose (they are supplied separately); relative terms like "the
relative major" or "a slower middle section" are fine. Write as instructions, not as analysis."""


def render_audio(midi_path, out_dir, soundfont=DEFAULT_SOUNDFONT, sample_rate=22050):
    """Render with FluidSynth, then compress to mono MP3 if ffmpeg is available."""
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, os.path.splitext(os.path.basename(midi_path))[0])
    mp3, wav = stem + ".mp3", stem + ".wav"
    if os.path.exists(mp3):
        return mp3
    subprocess.run(["fluidsynth", "-ni", "-F", wav, "-r", str(sample_rate), soundfont, midi_path],
                   check=True, capture_output=True)
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-ac", "1", "-b:a", "64k", mp3], check=True)
        os.remove(wav)
        return mp3
    return wav


def describe(midi_path, client, model, audio_dir):
    features = extract_features(midi_path)
    audio = client.files.upload(file=render_audio(midi_path, audio_dir))
    while audio.state.name == "PROCESSING":
        time.sleep(2)
        audio = client.files.get(name=audio.name)
    response = client.models.generate_content(
        model=model,
        contents=[PROMPT, json.dumps(features), audio],
        config={"response_mime_type": "application/json"},
    )
    client.files.delete(name=audio.name)
    desc = json.loads(response.text)
    missing = [k for k in VARIANTS + ("sections",) if k not in desc]
    if missing:
        raise ValueError(f"response missing {missing}")
    return {"file": os.path.basename(midi_path), **desc}


def fallback_description(midi_path):
    """Metadata-only description for smoke-testing the pipeline without an API key."""
    f = extract_features(midi_path)
    meta = f["metadata"]
    composer = next((name for m in meta for key, name in COMPOSERS.items() if key in m.lower()), None)
    marking = next((m for m in meta if any(w in m.lower() for w in TEMPO_WORDS) and len(m) < 40), "moderato")
    brief = f"Write a solo piano piece of about {f['n_bars']} bars, marked {marking}."
    styled = brief[:-1] + (f", in the style of {composer}." if composer else ".")
    return {
        "file": os.path.basename(midi_path),
        "brief": brief, "paragraph": brief, "detailed": brief,
        "styled_brief": styled, "styled_paragraph": styled, "styled_detailed": styled,
        "sections": [],
        "fallback": True,
    }


def describe_all(midi_paths, out_jsonl, client=None, model=None, audio_dir=None, retries=3):
    """Append one JSON line per piece; already-described files are skipped, so this resumes."""
    done = set()
    if os.path.exists(out_jsonl):
        with open(out_jsonl) as fh:
            done = {json.loads(line)["file"] for line in fh if line.strip()}
    for i, path in enumerate(midi_paths):
        name = os.path.basename(path)
        if name in done:
            continue
        for attempt in range(retries):
            try:
                rec = describe(path, client, model, audio_dir) if client else fallback_description(path)
                break
            except Exception as e:
                print(f"{name}: attempt {attempt + 1} failed: {e}")
                time.sleep(5 * (attempt + 1))
        else:
            continue
        with open(out_jsonl, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[{i + 1}/{len(midi_paths)}] {name}")
