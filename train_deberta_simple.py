#!/usr/bin/env python
"""
Minimal DeBERTa training for BC5CDR with direct execution.
No module-level imports that may hang; load incrementally.
"""
import sys
import os

# Suppress aiohttp import during datasets init by pre-loading selectively
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Direct imports to avoid hanging chains
print("Loading core dependencies...", flush=True)
import json
import csv
import random
import numpy as np
import torch
from seqeval.metrics import f1_score, precision_score, recall_score

print("Loading transformers & datasets...", flush=True)
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
from datasets import load_dataset, DatasetDict

print("Loading custom modules...", flush=True)
from src.config import ExperimentConfig

print("All imports successful.", flush=True)

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

def convert_kb_to_ner_bio(examples):
    """Convert KB format passages+entities to token-level BIO."""
    all_tokens = []
    all_ner_tags = []
    
    for passages, entities in zip(examples["passages"], examples["entities"]):
        full_text = " ".join([p["text"][0] for p in passages])
        tokens = full_text.split()
        
        token_spans = []
        char_idx = 0
        for token in tokens:
            start = full_text.find(token, char_idx)
            end = start + len(token)
            token_spans.append((start, end))
            char_idx = end
        
        tags = ["O"] * len(tokens)
        
        for ent in entities:
            ent_type = ent["type"]
            offsets = ent["offsets"]
            
            for offset_pair in offsets:
                ent_start, ent_end = offset_pair
                
                for idx, (tok_start, tok_end) in enumerate(token_spans):
                    if tok_start < ent_end and tok_end > ent_start:
                        if tags[idx] == "O":
                            if tok_start >= ent_start:
                                if idx > 0 and tags[idx-1].endswith(ent_type):
                                    tags[idx] = f"I-{ent_type}"
                                else:
                                    tags[idx] = f"B-{ent_type}"
                            else:
                                tags[idx] = f"B-{ent_type}"
        
        all_tokens.append(tokens)
        all_ner_tags.append(tags)
    
    return {"tokens": all_tokens, "ner_tags": all_ner_tags}

def tokenize_and_align_labels(examples, tokenizer, label_list, label2id, id2label, max_length=256):
    """Tokenize and align labels."""
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding=False,
        max_length=max_length,
    )

    labels = []
    for i, word_labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
                continue
            if word_id != prev_word_id:
                lab = word_labels[word_id]
                if isinstance(lab, str):
                    lab_id = label2id[lab]
                else:
                    lab_id = lab
                label_ids.append(lab_id)
            else:
                label_ids.append(-100)
            prev_word_id = word_id
        labels.append(label_ids)

    tokenized["labels"] = labels
    return tokenized

def compute_metrics_builder(label_list):
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
    return compute_metrics

# Main training
cfg = ExperimentConfig()
model_name = "microsoft/deberta-base"

print(f"\n===== Training {model_name} =====", flush=True)

# Load dataset
print("Loading BC5CDR dataset...", flush=True)
raw = load_dataset(cfg.dataset_name, cfg.dataset_config, trust_remote_code=True)

# Convert to BIO
print("Converting to BIO format...", flush=True)
ner_dict = {}
for split_name, ds in raw.items():
    ner_dict[split_name] = convert_kb_to_ner_bio({"passages": [ex["passages"] for ex in ds], "entities": [ex["entities"] for ex in ds]})
ner_dataset = DatasetDict(ner_dict)

# Extract labels
all_labels = set()
for split_ds in ner_dataset.values():
    for ex in split_ds:
        all_labels.update(ex["ner_tags"])
label_list = sorted(all_labels)
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for i, l in enumerate(label_list)}

# Load tokenizer (add_prefix_space=False for DeBERTa)
print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, add_prefix_space=False)

# Tokenize
print("Tokenizing datasets...", flush=True)
def tokenize_fn(batch):
    all_tokens = batch["tokens"]
    all_ner_tags = batch["ner_tags"]
    return tokenize_and_align_labels(
        {"tokens": all_tokens, "ner_tags": all_ner_tags},
        tokenizer, label_list, label2id, id2label, cfg.max_length
    )

tokenized = ner_dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=ner_dataset["train"].column_names
)
data_collator = DataCollatorForTokenClassification(tokenizer)

# Load model (FP32, PyTorch weights)
print("Loading model (FP32)...", flush=True)
model_cfg = AutoConfig.from_pretrained(
    model_name,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
)
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    config=model_cfg,
    use_safetensors=False,  # Use PyTorch .bin weights
)

out_dir = os.path.join(cfg.output_dir, model_name.replace("/", "--"))
os.makedirs(out_dir, exist_ok=True)

# Force FP32 for DeBERTa
fp16 = False

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

seed_everything(cfg.seed)

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=cfg.early_stopping_patience,
    early_stopping_threshold=0.0,
)

print("Creating trainer...", flush=True)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics_builder(label_list),
    callbacks=[early_stopping],
)

print("Starting training...", flush=True)
trainer.train()
eval_metrics = trainer.evaluate(tokenized["test"])

print(f"Evaluation metrics: {eval_metrics}", flush=True)

# Save metrics
with open(os.path.join(out_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(eval_metrics, f, indent=2)

result = {
    "model_name": model_name,
    "precision": float(eval_metrics.get("eval_precision", 0.0)),
    "recall": float(eval_metrics.get("eval_recall", 0.0)),
    "f1": float(eval_metrics.get("eval_f1", 0.0)),
    "output_dir": out_dir,
}
print(f"Result: {result}", flush=True)

# Append to metrics CSV
csv_path = cfg.results_csv
os.makedirs(os.path.dirname(csv_path), exist_ok=True)

# Check if we need to write header
file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
with open(csv_path, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["model_name", "precision", "recall", "f1", "output_dir"])
    if not file_exists:
        writer.writeheader()
    writer.writerow(result)

print(f"Saved to {csv_path}", flush=True)
