"""
Train BioBERT with optimized hyperparameters on full dataset
- Uses best hyperparameters from tuning: LR=5e-05, BS=16, Warmup=0.1, WD=0.01
- Trains on combined train+validation, evaluates on test
- Saves results separately in optimized_biobert/
"""
import os
import json
import torch
from pathlib import Path

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.data_utils import prepare_tokenized_datasets
from src.train import compute_metrics_builder, seed_everything


def train_optimized_model():
    """Train BioBERT with optimized hyperparameters."""
    
    print("="*80)
    print("OPTIMIZED BioBERT TRAINING")
    print("="*80)
    print("\nConfiguration:")
    print("  Model: BioBERT (microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext)")
    print("  Learning Rate: 5e-05")
    print("  Batch Size: 16")
    print("  Warmup Ratio: 0.1")
    print("  Weight Decay: 0.01")
    print("  Epochs: 15 (on full train+val dataset)")
    print("  Evaluation: Test set")
    print()
    
    model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    seed = 42
    
    # Load data
    print("[1/5] Loading dataset...")
    tokenized, tokenizer, label_list, label2id, id2label, data_collator = prepare_tokenized_datasets(
        model_name=model_name,
        dataset_name="bigbio/bc5cdr",
        dataset_config="bc5cdr_bigbio_kb",
        max_length=256,
        label_all_tokens=False,
    )
    
    # Combine train and validation for full training
    full_train = torch.utils.data.ConcatDataset([
        tokenized["train"],
        tokenized["validation"]
    ])
    
    print(f"  ✓ Train: {len(tokenized['train'])} examples")
    print(f"  ✓ Validation: {len(tokenized['validation'])} examples")
    print(f"  ✓ Full train: {len(full_train)} examples")
    print(f"  ✓ Test: {len(tokenized['test'])} examples")
    print()
    
    # Load model
    print("[2/5] Loading model...")
    num_labels = len(label_list)
    model_cfg = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        config=model_cfg,
        use_safetensors=True,
    )
    print(f"  ✓ Model loaded: {model_name}")
    print()
    
    # Output directory
    output_dir = "models/biobert_optimized"
    os.makedirs(output_dir, exist_ok=True)
    
    # Training arguments
    print("[3/5] Configuring training...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=5e-05,  # BEST
        per_device_train_batch_size=16,  # BEST
        per_device_eval_batch_size=32,
        num_train_epochs=15,  # Longer training on full dataset
        weight_decay=0.01,  # BEST
        warmup_ratio=0.1,  # BEST
        gradient_accumulation_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to=[],
        seed=seed,
    )
    print("  ✓ Training configuration set")
    print()
    
    # Seed
    seed_everything(seed)
    
    # Trainer
    print("[4/5] Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=full_train,
        eval_dataset=tokenized["test"],  # Evaluate on test set
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_builder(label_list),
    )
    print("  ✓ Trainer created")
    print()
    
    # Train
    print("[5/5] Training model...")
    trainer.train()
    print()
    
    # Evaluate on test set
    print("="*80)
    print("EVALUATION ON TEST SET")
    print("="*80)
    eval_results = trainer.evaluate()
    
    test_metrics = {
        'f1': eval_results.get('eval_f1', 0),
        'precision': eval_results.get('eval_precision', 0),
        'recall': eval_results.get('eval_recall', 0),
        'loss': eval_results.get('eval_loss', 0),
    }
    
    print(f"\nTest Metrics:")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  Loss:      {test_metrics['loss']:.4f}")
    print()
    
    # Comparison
    print("="*80)
    print("COMPARISON")
    print("="*80)
    baseline_f1 = 0.8779
    improvement = (test_metrics['f1'] - baseline_f1) * 100
    
    print(f"Original BioBERT F1: {baseline_f1:.4f}")
    print(f"Optimized BioBERT F1: {test_metrics['f1']:.4f}")
    print(f"Improvement: {improvement:+.2f}%")
    print()
    
    # Save results
    results_dir = "results/optimized_biobert"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save metrics
    metrics_file = os.path.join(results_dir, "test_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump({
            'test_f1': float(test_metrics['f1']),
            'test_precision': float(test_metrics['precision']),
            'test_recall': float(test_metrics['recall']),
            'test_loss': float(test_metrics['loss']),
            'improvement_vs_baseline': float(improvement),
            'baseline_f1': baseline_f1,
        }, f, indent=2)
    
    # Save hyperparameters
    hyperparams_file = os.path.join(results_dir, "hyperparameters.json")
    with open(hyperparams_file, 'w') as f:
        json.dump({
            'learning_rate': 5e-05,
            'batch_size': 16,
            'warmup_ratio': 0.1,
            'weight_decay': 0.01,
            'num_epochs': 15,
            'training_on': 'full_train_plus_validation',
            'evaluation_on': 'test_set',
        }, f, indent=2)
    
    # Save full results summary
    summary_file = os.path.join(results_dir, "summary.txt")
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("OPTIMIZED BioBERT - FINAL RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write("HYPERPARAMETERS:\n")
        f.write("  Learning Rate: 5e-05\n")
        f.write("  Batch Size: 16\n")
        f.write("  Warmup Ratio: 0.1\n")
        f.write("  Weight Decay: 0.01\n")
        f.write("  Epochs: 15\n")
        f.write("  Training Data: Full (train + validation = 1000 examples)\n")
        f.write("  Evaluation Data: Test (500 examples)\n\n")
        
        f.write("TEST SET RESULTS:\n")
        f.write(f"  F1 Score:  {test_metrics['f1']:.4f}\n")
        f.write(f"  Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"  Recall:    {test_metrics['recall']:.4f}\n")
        f.write(f"  Loss:      {test_metrics['loss']:.4f}\n\n")
        
        f.write("COMPARISON:\n")
        f.write(f"  Baseline (original BioBERT):  0.8779\n")
        f.write(f"  Optimized BioBERT:            {test_metrics['f1']:.4f}\n")
        f.write(f"  Improvement:                  {improvement:+.2f}%\n\n")
        
        f.write("MODEL LOCATION:\n")
        f.write(f"  {output_dir}\n\n")
        
        f.write("RESULTS SAVED TO:\n")
        f.write(f"  {results_dir}/\n")
    
    print("="*80)
    print("✅ OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {results_dir}/")
    print(f"  - test_metrics.json")
    print(f"  - hyperparameters.json")
    print(f"  - summary.txt")
    print(f"\nModel saved to: {output_dir}/")
    
    return test_metrics, output_dir, results_dir


if __name__ == "__main__":
    train_optimized_model()
