# Event classifier adapter (LoRA)

LoRA adapter (r=16, alpha=32, query/value) plus the trained 13-class head for `xlm-roberta-base`, produced by
`colab/train_classifier.ipynb` on a Colab T4 on 13 Nov 2026.

- Training data: 702 LLM-distilled headlines (`data/classifier/train.jsonl`, 78 more held out for validation).
- Test: 52 hand-written headlines (`data/fixtures/classifier_test.jsonl`): accuracy 0.808, macro-F1 0.800.
- Trainable parameters: 1,190,413 of 279,244,058 (0.43%).
- `adapter_model.safetensors` sha256: `392798423afa3e162f73f7236912f6617ae896ce0e3b31bcacf538cd83f736c1`
- The tokenizer is the base model's; it is not duplicated here.

Use: `pip install -r requirements-lora.txt` and set `RADAR_CLASSIFIER_MODEL=models/event-classifier-lora`.
