"""Optional fine-tuned tier for the event classifier: xlm-roberta-base + the LoRA adapter trained in
colab/train_classifier.ipynb.

It is optional on purpose. The base install has no PyTorch; set RADAR_CLASSIFIER_MODEL to an adapter folder (or a
Hugging Face repo id) and `pip install -r requirements-lora.txt` to switch it on. If anything is missing, `get()`
returns None and the classifier chain simply skips this tier.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from radar.config import ROOT, settings

log = logging.getLogger(__name__)


class LoraClassifier:
    def __init__(self, adapter: str):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = Path(adapter) if Path(adapter).exists() else ROOT / adapter
        source = str(path) if path.exists() else adapter  # local folder, else treat as a Hub id
        cfg = json.loads((path / "adapter_config.json").read_text(encoding="utf-8")) if path.exists() else {}
        base = cfg.get("base_model_name_or_path", "xlm-roberta-base")
        mapping = path / "label_mapping.json"
        if mapping.exists():
            self.labels = [v for _, v in sorted(json.loads(mapping.read_text(encoding="utf-8"))["id2label"].items(), key=lambda kv: int(kv[0]))]
        else:
            from radar.data.schemas import Topic

            self.labels = sorted(t.value for t in Topic)
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(base)
        model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=len(self.labels))
        self.model = PeftModel.from_pretrained(model, source)
        self.model.eval()

    def predict(self, titles: list[str]) -> list[tuple[str, float]]:
        """Returns (label, probability) per title."""
        torch = self._torch
        out: list[tuple[str, float]] = []
        with torch.no_grad():
            for i in range(0, len(titles), 32):
                enc = self.tok(titles[i: i + 32], truncation=True, max_length=64, padding=True, return_tensors="pt")
                probs = torch.softmax(self.model(**enc).logits, dim=-1)
                conf, idx = probs.max(dim=-1)
                out += [(self.labels[int(j)], float(c)) for j, c in zip(idx, conf, strict=True)]
        return out


@lru_cache(maxsize=1)
def get() -> LoraClassifier | None:
    if not settings.classifier_model:
        return None
    try:
        return LoraClassifier(settings.classifier_model)
    except Exception as e:  # missing torch/peft, missing files, download failure
        log.warning("LoRA classifier unavailable (%s: %s); continuing without it", type(e).__name__, str(e)[:120])
        return None
