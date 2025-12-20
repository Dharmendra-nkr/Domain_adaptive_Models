"""
Emergency DeBERTa training script with full offline mode.
Forces local_files_only everywhere to avoid HF Hub timeouts.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from src.config import ExperimentConfig
from src.data_utils import load_bc5cdr_ner, convert_kb_to_ner_bio, tokenize_and_align_labels
from datasets import DatasetDict
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
import torch
import json
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score

cfg = ExperimentConfig()
model_name = "microsoft/deberta-base"

# 1. Load dataset (offline)
os.environ["HF_DATASETS_OFFLINE"] = "1"
raw = load_bc5cdr_ner(cfg.dataset_name, cfg.dataset_config)

# 2. Convert KB to BIO
ner_dict = {}
for split_name, ds in raw.items():
    ner_dict[split_name] = convert_kb_to_ner_bio(ds)
ner_dataset = DatasetDict(ner_dict)

# Extract unique labels
all_labels = set()
for split_ds in ner_dataset.values():
    for ex in split_ds:
        all_labels.update(ex["ner_tags"])
label_list = sorted(all_labels)
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

# 3. Tokenizer (offline, no template check)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=True,
    add_prefix_space=False,  # DeBERTa doesn't need it
    local_files_only=True,
)

# 4. Tokenize
def tokenize_fn(examples):
    return tokenize_and_align_labels(
        examples,
        tokenizer,
        label_list,
        max_length=cfg.max_length,
        label_all_tokens=cfg.label_all_tokens,
    )

tokenized = ner_dataset.map(tokenize_fn, batched=True, remove_columns=ner_dataset["train"].column_names)
data_collator = DataCollatorForTokenClassification(tokenizer)

# 5. Model (offline, no safetensors auto-convert)
model_cfg = AutoConfig.from_pretrained(
    model_name,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
    local_files_only=True,
)
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    config=model_cfg,
    use_safetensors=False,  # Use PyTorch .bin
    local_files_only=True,
)

out_dir = os.path.join(cfg.output_dir, "microsoft--deberta-base")
os.makedirs(out_dir, exist_ok=True)

# DeBERTa FP16 overflow fix
fp16 = False  # Force FP32 for DeBERTa

args = TrainingArguments(
    output_dir=out_dir,
    learning_rate=cfg.learning_rate,
    num_train_epochs=cfg.num_train_epochs,
    weight_decay=cfg.weight_decay,
    per_device_train_batch_size=cfg.per_device_train_batch_size,
    per_device_eval_batch_size=cfg.per_device_eval_batch_size,
    gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    warmup_ratio=cfg.warmup_ratio,
    eval_strategy=cfg.evaluation_strategy,
    save_strategy=cfg.save_strategy,
    load_best_model_at_end=cfg.load_best_model_at_end,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=cfg.logging_steps,
    fp16=fp16,
    report_to=list(cfg.report_to),
    save_total_limit=1,
)

def compute_metrics(p):
    preds, labels = p
    preds = np.argmax(preds, axis=2)
    true_predictions = []
    true_labels = []
    for pred, lab in zip(preds, labels):
        valid_idx = lab != -100
        pred = pred[valid_idx]
        lab = lab[valid_idx]
        true_predictions.append([label_list[p] for p in pred])
        true_labels.append([label_list[l] for l in lab])
    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=cfg.early_stopping_patience,
    early_stopping_threshold=0.0,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping],
)

print("Starting DeBERTa training (FP32, offline mode)...")
trainer.train()
eval_metrics = trainer.evaluate(tokenized["test"])

with open(os.path.join(out_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(eval_metrics, f, indent=2)

result = {
    "model_name": model_name,
    "precision": float(eval_metrics.get("eval_precision", 0.0)),
    "recall": float(eval_metrics.get("eval_recall", 0.0)),
    "f1": float(eval_metrics.get("eval_f1", 0.0)),
    "output_dir": out_dir,
}
print(f"Result: {result}")

# Append to metrics CSV
import csv
csv_path = cfg.results_csv
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
with open(csv_path, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["model_name", "precision", "recall", "f1", "output_dir"])
    if os.path.getsize(csv_path) == 0:
        writer.writeheader()
    writer.writerow(result)

print(f"Appended result to {csv_path}")
