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


def archive_stale_results(results_path, required_metric="notes_per_bar"):
    """Rename results.jsonl if any row predates the current metrics schema.

    Returns the archive path, or None if the file was current or missing.
    """
    rows = load_results(results_path)
    if not rows or all(required_metric in (r.get("metrics") or {}) for r in rows):
        return None
    dest = f"{os.path.splitext(results_path)[0]}_old_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    os.rename(results_path, dest)
    return dest


# Corpus-level rates the rerank aims at. A piece is kept when it lands near these,
# which is where the training MIDI actually sits, rather than at 0 or 1.
_RERANK = {
    "diatonic_frac": 0.85,
    "rhythm_repeat_frac": 0.2,
    "silence_frac": 0.05,
    "tempo_jump_frac": 0.1,
}


def _rerank_distance(metrics):
    """Distance from the corpus rates. A missing rate (no notes, or one bar) counts as a total miss."""
    total = 0.0
    for key, target in _RERANK.items():
        got = (metrics or {}).get(key)
        if isinstance(got, bool) or not isinstance(got, (int, float)):
            total += 1.0
        else:
            total += abs(float(got) - target)
    return total


def run_eval(generate_fn, files, texts, descs, out_dir, tag, seeds=(0,), variant="styled_detailed",
             max_chunks=40, train_lines=None, set_seed=None, settings=None, render=True, on_piece=None,
             select_best=False):
    """Compose from each file's brief for each seed, save outputs, and append results.

    Length is the original piece's bar count, so a generated prelude is as long as
    the prelude. Pieces already logged for this tag and seed are skipped, so an
    interrupted run resumes. With `select_best`, every seed is generated and only
    the closest to the corpus rates is logged; the other seeds stay under `candidates`.
    `set_seed(seed)` is called before each piece so sampling is reproducible.
    """
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "results.jsonl")
    archived = archive_stale_results(results_path)
    if archived:
        print(f"Archived old-format results to {archived}")
    logged = load_results(results_path)
    done = {(r["file"], r["seed"], r["tag"]) for r in logged}
    done_files = {(r["file"], r["tag"]) for r in logged}
    rows = []
    for f in files:
        if f not in descs or f not in texts:
            continue
        if select_best and (f, tag) in done_files:
            continue
        brief = descs[f].get(variant) or descs[f]["brief"]
        meta = meta_fields(texts[f].splitlines()[1])
        key, meter = meta.get("key"), meta.get("ts")
        n_bars = int(meta.get("bars") or 0) or None
        reference = None
        pending = []
        for seed in seeds:
            if not select_best and (f, seed, tag) in done:
                continue
            if set_seed:
                set_seed(seed)
            start = time.time()
            text = compose(generate_fn, brief, key=key, meter=meter, bars=n_bars, max_chunks=max_chunks)
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
            pending.append(row)
        if not pending:
            continue
        if select_best and len(pending) > 1:
            pending.sort(key=lambda r: _rerank_distance(r["metrics"]))
            winner = pending[0]
            winner["candidates"] = [
                {"seed": r["seed"], "metrics": r["metrics"], "paths": r["paths"]} for r in pending[1:]
            ]
            pending = [winner]
        for row in pending:
            with open(results_path, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            rows.append(row)
            if on_piece:
                on_piece(row)
    return rows
