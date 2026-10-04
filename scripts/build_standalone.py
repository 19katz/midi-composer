"""Build notebooks/midi_composer_colab.ipynb: pipeline.ipynb with the midicomposer package
inlined as %%writefile cells, so the notebook runs in Colab on its own (upload and run).

Usage: python scripts/build_standalone.py [--check]
"""

import argparse
import os
import sys

import nbformat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "notebooks", "pipeline.ipynb")
TARGET = os.path.join(ROOT, "notebooks", "midi_composer_colab.ipynb")
PKG = os.path.join(ROOT, "src", "midicomposer")
MODULES = ["__init__", "codec", "features", "describe", "dataset", "compose", "evaluate", "runs"]
LIB = "midicomposer_lib"

INTRO = """**Standalone version.** Upload this file to Colab (File > Upload notebook) and run it; nothing else is needed. The library code is in the "Library code" section below. It is generated from `src/midicomposer/` in the repo by `scripts/build_standalone.py`, so make code changes there.

"""

SETUP = f"""import glob, json, os, sys

IN_COLAB = 'google.colab' in sys.modules
USE_DRIVE = False
REPO_DIR = os.getcwd()

if IN_COLAB:
    from google.colab import drive
    try:
        drive.mount('/content/drive')
        USE_DRIVE = True
    except Exception as e:
        print(f'Google Drive did not mount: {{e}}\\n'
              'Continuing with storage local to this runtime; it is lost when the runtime disconnects.\\n'
              'To use Drive: rerun this cell, pick the same Google account as Colab, and tick every permission.')
    !pip install -q pretty_midi mido google-genai datasets
    !apt-get -qq install -y fluidsynth fluid-soundfont-gm > /dev/null

os.makedirs('{LIB}/midicomposer', exist_ok=True)
sys.path.insert(0, os.path.abspath('{LIB}'))"""

LIB_INTRO = """## Library code

The `midicomposer` package, written to `midicomposer_lib/` so the cells below can import it. Run these cells; don't edit them here (they are regenerated from the repo's `src/midicomposer/`)."""


def build():
    nb = nbformat.read(SOURCE, as_version=4)
    cells = nb.cells
    assert cells[0].cell_type == "markdown" and "Before running in Colab" in cells[0].source
    assert cells[1].cell_type == "code" and "REPO_URL" in cells[1].source

    intro = nbformat.v4.new_markdown_cell(cells[0].source.replace("**Before running in Colab**", INTRO + "**Before running**", 1))
    lib = [nbformat.v4.new_markdown_cell(LIB_INTRO)]
    for name in MODULES:
        with open(os.path.join(PKG, f"{name}.py")) as fh:
            src = fh.read().rstrip("\n")
        lib.append(nbformat.v4.new_code_cell(f"%%writefile {LIB}/midicomposer/{name}.py\n{src}\n"))
    lib.append(nbformat.v4.new_code_cell("import importlib\nimportlib.invalidate_caches()"))

    out = nbformat.v4.new_notebook(metadata=nb.metadata)
    out.cells = [intro, nbformat.v4.new_code_cell(SETUP), *lib, *cells[2:]]
    for i, c in enumerate(out.cells):
        c["id"] = f"cell-{i:02d}"
        if c.cell_type == "code":
            c.outputs, c.execution_count = [], None
    nbformat.validate(out)
    return nbformat.writes(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if the committed notebook is stale")
    args = ap.parse_args()
    text = build()
    if args.check:
        current = open(TARGET).read() if os.path.exists(TARGET) else ""
        if current != text:
            sys.exit(f"{os.path.relpath(TARGET, ROOT)} is stale; run scripts/build_standalone.py")
        return
    with open(TARGET, "w") as fh:
        fh.write(text)
    print(f"wrote {os.path.relpath(TARGET, ROOT)}")


if __name__ == "__main__":
    main()
