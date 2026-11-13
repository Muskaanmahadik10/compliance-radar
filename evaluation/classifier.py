"""Classifier evaluation on a hand-written test set of realistic headlines (data/fixtures/classifier_test.jsonl).

Compares the tiers on macro-F1 and accuracy:
  rules      keyword classifier (free, instant)
  rules+llm  rules, with the cheap LLM deciding low-confidence items (needs a provider; skipped under mock)
  lora       fine-tuned encoder from colab/train_classifier.ipynb, if RADAR_CLASSIFIER_MODEL points at it

Usage: python -m evaluation.classifier
"""

from __future__ import annotations

import json
import time
from collections import Counter

from radar.config import ARTIFACTS_DIR, DATA_DIR, settings
from radar.data.schemas import Topic
from radar.ingest.classify import classify, classify_rules, llm_callback
from radar.llm import CostLedger, get_router


def load() -> list[dict]:
    p = DATA_DIR / "fixtures" / "classifier_test.jsonl"
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    labels = sorted(set(y_true))
    f1s = []
    for lab in labels:
        tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred, strict=True))
        fp = sum(t != lab and p == lab for t, p in zip(y_true, y_pred, strict=True))
        fn = sum(t == lab and p != lab for t, p in zip(y_true, y_pred, strict=True))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)


def _row(name: str, y: list[str], pred: list[str], extra: dict | None = None) -> dict:
    wrong = Counter((t, p) for t, p in zip(y, pred, strict=True) if t != p)
    return {"tier": name, "n": len(y), "accuracy": round(sum(t == p for t, p in zip(y, pred, strict=True)) / len(y), 3), "macro_f1": round(macro_f1(y, pred), 3),
            "top_confusions": [f"{t}->{p} x{n}" for (t, p), n in wrong.most_common(4)]} | (extra or {})


def run() -> dict:
    data = load()
    y = [d["topic"] for d in data]
    titles = [d["title"] for d in data]
    rows = [_row("rules", y, [classify_rules(t).topic.value for t in titles], {"cost_usd": 0.0, "llm_calls": 0})]

    # Fine-tuned tier, if the adapter and its libraries are available (RADAR_CLASSIFIER_MODEL).
    from radar.ingest import lora_classifier

    lora = lora_classifier.get()
    if lora is not None:
        t0 = time.perf_counter()
        preds = lora.predict(titles)
        ms = (time.perf_counter() - t0) * 1000 / len(titles)
        rows.append(_row("LoRA encoder alone", y, [p for p, _ in preds], {"cost_usd": 0.0, "llm_calls": 0, "ms_per_headline_cpu": round(ms, 1)}))
        thr = settings.classifier_lora_min_confidence
        chain = [classify(t, lora=lora, lora_min_confidence=thr) for t in titles]
        rows.append(_row(f"rules -> LoRA (no LLM, threshold {thr})", y, [c.topic.value for c in chain], {"cost_usd": 0.0, "llm_calls": 0}))

    router = get_router()
    if router.cheap._llm_type != "mock":
        def run_chain(name: str, use_lora):
            ledger = CostLedger()
            llm = llm_callback(router, callbacks=[ledger])  # the same callable the live worker uses

            # _Off() disables the tier explicitly (passing None would auto-load a configured adapter)
            out = [classify(t, llm=llm, lora=use_lora if use_lora is not None else _Off()) for t in titles]
            tiers = Counter(c.tier for c in out)
            rows.append(_row(name, y, [c.topic.value for c in out], {"cost_usd": round(ledger.cost_usd, 5), "llm_calls": ledger.calls, "decided_by": dict(tiers)}))

        run_chain("rules -> LLM", None)
        if lora is not None:
            run_chain(f"rules -> LoRA -> LLM (threshold {settings.classifier_lora_min_confidence})", lora)
    out = {"labels": [t.value for t in Topic], "n": len(data), "results": rows}
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "classifier_eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


class _Off:
    """Stand-in that makes the LoRA tier abstain, so 'rules -> LLM' can be measured even when an adapter is configured."""

    def predict(self, titles):
        raise RuntimeError("LoRA tier disabled for this measurement")  # classify() treats a failing tier as absent


if __name__ == "__main__":
    for r in run()["results"]:
        print(json.dumps(r))
