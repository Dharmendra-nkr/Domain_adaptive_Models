"""
Evaluation utilities: load a trained model dir and compute metrics on BC5CDR test.
"""
import json
import os
from typing import Dict, List

import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer

from .config import ExperimentConfig
from .data_utils import prepare_tokenized_datasets


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


def evaluate_model_dir(model_dir: str, cfg: ExperimentConfig) -> Dict:
    tokenized, tokenizer, label_list, label2id, id2label, data_collator = prepare_tokenized_datasets(
        model_name=model_dir,
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        max_length=cfg.max_length,
        label_all_tokens=cfg.label_all_tokens,
    )

    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_builder(label_list),
    )

    metrics = trainer.evaluate(tokenized["test"])
    with open(os.path.join(model_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


if __name__ == "__main__":
    cfg = ExperimentConfig()
    print(evaluate_model_dir(os.path.join(cfg.output_dir, cfg.model_names[0].replace("/", "--")), cfg))
