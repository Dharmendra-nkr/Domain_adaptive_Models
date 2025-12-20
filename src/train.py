"""
Training script for biomedical NER with HF Transformers Trainer.
- Uses identical hyperparameters for all models
- Mixed-precision (fp16) on GPU when available
- Saves best model and evaluation metrics
"""
import os
import json
import random
from dataclasses import asdict
from typing import Dict, List

import numpy as np
import torch
from datasets import DatasetDict
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .data_utils import prepare_tokenized_datasets
from .config import ExperimentConfig


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def compute_metrics_builder(label_list: List[str]):
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


def train_single_model(model_name: str, cfg: ExperimentConfig) -> Dict:
    tokenized, tokenizer, label_list, label2id, id2label, data_collator = prepare_tokenized_datasets(
        model_name=model_name,
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        max_length=cfg.max_length,
        label_all_tokens=cfg.label_all_tokens,
    )

    num_labels = len(label_list)
    model_cfg = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    # DeBERTa may need PyTorch weights if safetensors unavailable
    use_st = "deberta" not in model_name.lower()
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        config=model_cfg,
        use_safetensors=use_st,
    )

    out_dir = os.path.join(cfg.output_dir, model_name.replace("/", "--"))
    os.makedirs(out_dir, exist_ok=True)

    # DeBERTa FP16 overflow bug fix: disable FP16 for DeBERTa models
    fp16 = cfg.fp16 and torch.cuda.is_available() and "deberta" not in model_name.lower()

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
        save_total_limit=1,  # Keep only the best checkpoint to save disk space
    )

    seed_everything(cfg.seed)

    # Create early stopping callback
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
        compute_metrics=compute_metrics_builder(label_list),
        callbacks=[early_stopping],
    )

    trainer.train()
    eval_metrics = trainer.evaluate(tokenized["test"])  # evaluates best model

    # Save metrics
    with open(os.path.join(out_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)

    return {
        "model_name": model_name,
        "precision": float(eval_metrics.get("eval_precision", eval_metrics.get("precision", 0.0))),
        "recall": float(eval_metrics.get("eval_recall", eval_metrics.get("recall", 0.0))),
        "f1": float(eval_metrics.get("eval_f1", eval_metrics.get("f1", 0.0))),
        "output_dir": out_dir,
    }


if __name__ == "__main__":
    cfg = ExperimentConfig()
    res = train_single_model(cfg.model_names[0], cfg)
    print(res)
