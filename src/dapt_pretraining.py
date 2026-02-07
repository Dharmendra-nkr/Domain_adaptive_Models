"""
Domain-Adaptive Pre-Training (DAPT) for BERT on biomedical corpus.
Continues pre-training BERT-base using Masked Language Modeling on PubMed abstracts.
"""
import os
import sys
from dataclasses import asdict

import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .dapt_config import DAPTConfig
from .dapt_utils import (
    estimate_training_time,
    load_pubmed_corpus,
    prepare_mlm_dataset,
    print_dataset_stats,
)


def run_dapt(cfg: DAPTConfig = None):
    """
    Run Domain-Adaptive Pre-Training.
    
    Args:
        cfg: DAPT configuration (uses defaults if None)
    """
    if cfg is None:
        cfg = DAPTConfig()
    
    print("="*70)
    print("DOMAIN-ADAPTIVE PRE-TRAINING (DAPT)")
    print("="*70)
    print(f"Base Model: {cfg.base_model}")
    print(f"Output Directory: {cfg.output_dir}")
    print(f"Corpus: {cfg.corpus_name} ({cfg.corpus_subset or 'full'})")
    print(f"Epochs: {cfg.num_train_epochs}")
    print(f"Batch Size: {cfg.per_device_train_batch_size} (effective: {cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps})")
    print("="*70 + "\n")
    
    # Set seed for reproducibility
    set_seed(cfg.seed)
    
    # Load tokenizer and model
    print(f"Loading tokenizer and model: {cfg.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForMaskedLM.from_pretrained(cfg.base_model)
    
    print(f"Model parameters: {model.num_parameters():,}")
    
    # Load and prepare corpus
    print(f"\nLoading corpus: {cfg.corpus_name}")
    raw_dataset = load_pubmed_corpus(
        corpus_name=cfg.corpus_name,
        subset=cfg.corpus_subset,
        validation_split=cfg.validation_split_percentage / 100.0
    )
    
    print_dataset_stats(raw_dataset)
    
    # Prepare for MLM
    tokenized_dataset = prepare_mlm_dataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        max_length=cfg.max_length,
        num_workers=cfg.preprocessing_num_workers
    )
    
    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=cfg.mlm_probability
    )
    
    # Estimate training time
    effective_batch_size = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        effective_batch_size *= num_gpus
    
    total_seconds, time_str = estimate_training_time(
        num_examples=len(tokenized_dataset["train"]),
        batch_size=effective_batch_size,
        num_epochs=cfg.num_train_epochs,
        seconds_per_batch=0.5 if torch.cuda.is_available() else 2.0
    )
    
    print(f"\nEstimated training time: {time_str}")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        fp16=cfg.fp16 and torch.cuda.is_available(),
        logging_dir=os.path.join(cfg.output_dir, "logs"),
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        evaluation_strategy=cfg.evaluation_strategy,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=cfg.dataloader_num_workers,
        seed=cfg.seed,
        report_to=[],  # Disable wandb/tensorboard by default
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
    )
    
    # Train
    print("\n" + "="*70)
    print("Starting DAPT training...")
    print("="*70 + "\n")
    
    train_result = trainer.train()
    
    # Save final model
    print("\nSaving model to:", cfg.output_dir)
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Evaluate
    print("\nEvaluating on validation set...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    print("\n" + "="*70)
    print("DAPT COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Model saved to: {cfg.output_dir}")
    print(f"Training loss: {metrics['train_loss']:.4f}")
    print(f"Validation loss: {eval_metrics['eval_loss']:.4f}")
    print("\nNext steps:")
    print(f"1. Add '{cfg.output_dir}' to model_names in src/config.py")
    print("2. Run: python -m src.run_experiments --models", cfg.output_dir)
    print("="*70 + "\n")
    
    return trainer, metrics, eval_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run DAPT on biomedical corpus")
    parser.add_argument(
        "--base-model",
        type=str,
        default="bert-base-uncased",
        help="Base model to continue pre-training"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="pubmed",
        help="Corpus name"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="train[:100000]",
        help="Corpus subset (e.g., 'train[:100000]')"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Per-device batch size"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/bert-base-dapt",
        help="Output directory for DAPT model"
    )
    
    args = parser.parse_args()
    
    # Create config from args
    cfg = DAPTConfig(
        base_model=args.base_model,
        corpus_name=args.corpus,
        corpus_subset=args.subset,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
    
    run_dapt(cfg)
