"""Smoke-test the Colab training notebook end to end on CPU: python scripts/smoke_test_notebook.py

Runs every code cell of colab/train_classifier.ipynb in order, in a scratch folder, with three deliberate changes so it
finishes in minutes on a laptop: shell lines (!pip, !git, %cd) are skipped and the installed packages are used; the
training/validation sets are cut to 64/26 headlines; and training stops after 2 optimizer steps. The notebook's own
TrainingArguments are constructed first exactly as written, so an argument the installed transformers rejects (the
old `warmup_ratio`) fails this script. Save, manifest and the reload/label-mapping check run unchanged.

This proves the cells run with the recorded stack. The test scores it prints are meaningless (2 steps); the 0.808
accuracy in the README comes from the original 13 Nov Colab run.
Needs: torch, transformers, peft, accelerate, datasets, scikit-learn, and the xlm-roberta-base weights (downloaded
or cached). Writes artifacts/notebook_smoke_<date>.json.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "colab" / "train_classifier.ipynb"


def main() -> dict:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    work = Path(tempfile.mkdtemp(prefix="nb-smoke-"))
    for sub in ("classifier", "fixtures"):
        shutil.copytree(ROOT / "data" / sub, work / "data" / sub)
    os.chdir(work)
    ns: dict = {"__name__": "__notebook__"}
    validated = {}
    for i, cell in enumerate(c for c in nb["cells"] if c["cell_type"] == "code"):
        src = "".join(cell["source"])
        body = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith(("!", "%")))
        if "args = TrainingArguments(" in body:
            from transformers import TrainingArguments as _TA

            def smoke_args(**kw):
                _TA(**kw)  # the notebook's arguments exactly as written must be accepted
                validated.update({k: v for k, v in kw.items() if isinstance(v, (int, float, str, bool))})
                return _TA(**(kw | {"max_steps": 2, "eval_strategy": "no", "save_strategy": "no", "load_best_model_at_end": False, "logging_steps": 1}))

            body = body.replace("args = TrainingArguments(", "args = _smoke_args(")
            ns["_smoke_args"] = smoke_args
        exec(compile(body, f"notebook cell {i}", "exec"), ns)
        if "val_rows, train_rows = " in body:  # after the data cell: a small subset is enough to exercise the code
            ns["train_rows"], ns["val_rows"] = ns["train_rows"][:64], ns["val_rows"][:26]
        print(f"cell {i} ok")
    manifest = json.loads((work / "clf-adapter" / "run_manifest.json").read_text(encoding="utf-8"))
    record = {
        "what": "colab/train_classifier.ipynb executed cell by cell on CPU as a smoke test (scripts/smoke_test_notebook.py)",
        "changes_for_smoke": ["shell lines skipped; installed packages used", "train/validation cut to 64/26 headlines", "max_steps=2, no per-epoch eval/save"],
        "notebook_sha256": hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
        "repo_commit": subprocess.run(["git", "describe", "--always", "--dirty", "--abbrev=7"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        "notebook_training_arguments_accepted": validated,
        "reload_check": "passed (same label mapping, identical test predictions)",
        "manifest_written_by_notebook": manifest,
        "scores_note": "test scores after 2 steps are meaningless; the reported 0.808 accuracy is the original 13 Nov Colab run",
        "run_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }
    out = ROOT / "artifacts" / f"notebook_smoke_{dt.date.today().isoformat()}.json"
    out.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    os.chdir(ROOT)
    shutil.rmtree(work, ignore_errors=True)
    print("wrote", out.relative_to(ROOT))
    return record


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
