"""Reproduces the study by running the original script in code/.

code/experiment.py - corpus audit (exact duplicates and label conflicts), one stratified 80/20 split of the
unique unambiguous abstracts, training-only tuning of L1 and L2 logistic regression, held-out scoring, paired
bootstrap, vocabulary stability and fig_results.png.
Protocol fixed before modelling: code/protocol.md. Data: the Medical Abstracts Text Classification Corpus,
bundled in code/data/ under CC BY-SA 3.0 (code/data/LICENSE.txt). Requires Python 3.12 and
code/requirements.txt.
"""

import os
import runpy
import sys
from pathlib import Path

script = Path(__file__).resolve().parent / "code" / "experiment.py"
os.chdir(script.parent)
sys.path.insert(0, str(script.parent))
runpy.run_path(str(script), run_name="__main__")
