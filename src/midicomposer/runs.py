"""Generate, save, and log evaluation runs so checkpoints and settings can be compared later.

Layout under `out_dir`:
    results.jsonl                      one line per (file, seed, tag), appended as pieces finish
    <tag>/<piece>__<tag>__s<seed>.mid  generated MIDI (+ .txt score, .wav render if FluidSynth works)
"""

import json
import os
import subprocess
import time

from .codec import decode
from .compose import compose
from .dataset import meta_fields
from .describe import DEFAULT_SOUNDFONT
from .evaluate import evaluate


def render_wav(mid_path, wav_path, soundfont=DEFAULT_SOUNDFONT, sample_rate=22050):
    try:
        subprocess.run(["fluidsynth", "-ni", "-F", wav_path, "-r", str(sample_rate), soundfont, mid_path],
                       check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return wav_path if os.path.exists(wav_path) else None


def save_piece(text, out_dir, name, render=True):
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, name)
    with open(stem + ".txt", "w") as fh:
        fh.write(text)
    decode(text).save(stem + ".mid")
    wav = render_wav(stem + ".mid", stem + ".wav") if render else None
    return {"txt": stem + ".txt", "mid": stem + ".mid", "wav": wav}


def load_results(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run_eval(generate_fn, files, texts, descs, out_dir, tag, seeds=(0,), variant="styled_detailed",
             max_chunks=40, train_lines=None, set_seed=None, settings=None, render=True, on_piece=None):
    """Compose from each file's brief for each seed, save outputs, and append results.

    Pieces already logged for this tag and seed are skipped, so an interrupted run resumes.
    `set_seed(seed)` is called before each piece so sampling is reproducible.
    """
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "results.jsonl")
    done = {(r["file"], r["seed"], r["tag"]) for r in load_results(results_path)}
    rows = []
    for f in files:
        if f not in descs or f not in texts:
            continue
        brief = descs[f].get(variant) or descs[f]["brief"]
        meta = meta_fields(texts[f].splitlines()[1])
        key, meter = meta.get("key"), meta.get("ts")
        reference = None
        for seed in seeds:
            if (f, seed, tag) in done:
                continue
            if set_seed:
                set_seed(seed)
            start = time.time()
            text = compose(generate_fn, brief, key=key, meter=meter, max_chunks=max_chunks)
            seconds = round(time.time() - start, 1)
            if reference is None:
                reference = evaluate(texts[f], requested_key=key, train_lines=train_lines)
            row = {
                "tag": tag,
                "file": f,
                "seed": seed,
                "variant": variant,
                "brief": brief,
                "key": key,
                "meter": meter,
                "settings": settings or {},
                "max_chunks": max_chunks,
                "seconds": seconds,
                "metrics": evaluate(text, requested_key=key, reference_text=texts[f], train_lines=train_lines),
                "reference_metrics": reference,
                "paths": save_piece(text, os.path.join(out_dir, tag), f"{f[:-4]}__{tag}__s{seed}", render),
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(results_path, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            rows.append(row)
            if on_piece:
                on_piece(row)
    return rows
