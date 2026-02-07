"""
Hyperparameter Tuning for BioBERT on BC5CDR
- Systematic search over key hyperparameters
- Learning rate, batch size, warmup, weight decay
- Uses validation set for quick evaluation
"""
import os
import json
import itertools
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, List

import torch
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed
)

from src.data_utils import prepare_tokenized_datasets
from src.train import compute_metrics_builder, seed_everything


@dataclass
class HyperparamConfig:
    """Hyperparameter configuration for tuning."""
    learning_rate: float
    per_device_train_batch_size: int
    warmup_ratio: float
    weight_decay: float
    num_train_epochs: int = 10  # Shorter for tuning
    gradient_accumulation_steps: int = 1


def resolve_model_source(model_name: str) -> str:
    """Prefer local fine-tuned checkpoint if available to avoid network/torch.load issues."""
    local_checkpoint = os.path.join(
        "models",
        "microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "checkpoint-224",
    )
    if os.path.isdir(local_checkpoint):
        print(f"[INFO] Using local checkpoint: {local_checkpoint}")
        return local_checkpoint
    print(f"[INFO] Using Hugging Face model: {model_name}")
    return model_name


def train_with_hyperparams(config: HyperparamConfig, model_name: str, seed: int = 42) -> Dict:
    """Train model with specific hyperparameters and return validation metrics."""
    
    print(f"\n{'='*80}")
    print(f"Testing: LR={config.learning_rate}, BS={config.per_device_train_batch_size}, "
          f"Warmup={config.warmup_ratio}, WD={config.weight_decay}")
    print(f"{'='*80}")
    
    # Resolve model source
    model_source = resolve_model_source(model_name)

    # Load data
    tokenized, tokenizer, label_list, label2id, id2label, data_collator = prepare_tokenized_datasets(
        model_name=model_source,
        dataset_name="bigbio/bc5cdr",
        dataset_config="bc5cdr_bigbio_kb",
        max_length=256,
        label_all_tokens=False,
    )
    
    # Load model
    num_labels = len(label_list)
    model_cfg = AutoConfig.from_pretrained(
        model_source,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_source,
        config=model_cfg,
        use_safetensors=True,
        local_files_only=os.path.isdir(model_source),
    )
    
    # Output directory for this configuration
    output_dir = f"models/tuning_temp/lr{config.learning_rate}_bs{config.per_device_train_batch_size}_wd{config.weight_decay}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=32,
        num_train_epochs=config.num_train_epochs,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        eval_strategy="epoch",
        save_strategy="no",  # Don't save checkpoints during tuning
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to=[],
        load_best_model_at_end=False,
        seed=seed,
    )
    
    seed_everything(seed)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_builder(label_list),
    )
    
    # Train
    trainer.train()
    
    # Evaluate on validation set
    eval_results = trainer.evaluate()
    
    # Clean up model to save memory
    del model
    del trainer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return {
        'learning_rate': config.learning_rate,
        'batch_size': config.per_device_train_batch_size,
        'warmup_ratio': config.warmup_ratio,
        'weight_decay': config.weight_decay,
        'num_epochs': config.num_train_epochs,
        'val_precision': eval_results.get('eval_precision', 0),
        'val_recall': eval_results.get('eval_recall', 0),
        'val_f1': eval_results.get('eval_f1', 0),
        'val_loss': eval_results.get('eval_loss', 0),
    }


def grid_search():
    """Perform grid search over hyperparameter space."""
    
    print("="*80)
    print("HYPERPARAMETER TUNING - BioBERT on BC5CDR")
    print("="*80 + "\n")
    
    model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    
    # Define search space
    learning_rates = [1e-5, 2e-5, 3e-5, 5e-5]
    batch_sizes = [8, 16]
    warmup_ratios = [0.0, 0.1, 0.2]
    weight_decays = [0.0, 0.01, 0.1]
    
    print("SEARCH SPACE")
    print("="*80)
    print(f"Learning rates: {learning_rates}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Warmup ratios: {warmup_ratios}")
    print(f"Weight decays: {weight_decays}")
    print(f"\nTotal configurations: {len(learning_rates) * len(batch_sizes) * len(warmup_ratios) * len(weight_decays)}")
    print(f"Estimated time: ~{len(learning_rates) * len(batch_sizes) * len(warmup_ratios) * len(weight_decays) * 5} minutes")
    print()
    
    # Run grid search
    results = []
    config_num = 0
    total_configs = len(learning_rates) * len(batch_sizes) * len(warmup_ratios) * len(weight_decays)
    
    for lr, bs, warmup, wd in itertools.product(learning_rates, batch_sizes, warmup_ratios, weight_decays):
        config_num += 1
        print(f"\n{'#'*80}")
        print(f"Configuration {config_num}/{total_configs}")
        print(f"{'#'*80}")
        
        config = HyperparamConfig(
            learning_rate=lr,
            per_device_train_batch_size=bs,
            warmup_ratio=warmup,
            weight_decay=wd,
            num_train_epochs=10,
        )
        
        try:
            result = train_with_hyperparams(config, model_name)
            results.append(result)
            
            print(f"\n✓ Validation F1: {result['val_f1']:.4f}")
            print(f"  Precision: {result['val_precision']:.4f}")
            print(f"  Recall: {result['val_recall']:.4f}")
            
        except Exception as e:
            print(f"\n✗ Configuration failed: {e}")
            continue
    
    return results


def quick_search():
    """Quick search with most promising configurations."""
    
    print("="*80)
    print("QUICK HYPERPARAMETER SEARCH - BioBERT on BC5CDR")
    print("="*80 + "\n")
    
    model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    
    # Focused search on most impactful parameters
    configs = [
        # Baseline (current best)
        HyperparamConfig(2e-5, 16, 0.1, 0.01, 10),
        # Lower LR
        HyperparamConfig(1e-5, 16, 0.1, 0.01, 10),
        # Higher LR
        HyperparamConfig(3e-5, 16, 0.1, 0.01, 10),
        HyperparamConfig(5e-5, 16, 0.1, 0.01, 10),
        # Different batch sizes
        HyperparamConfig(2e-5, 8, 0.1, 0.01, 10),
        HyperparamConfig(2e-5, 32, 0.1, 0.01, 10),
        # Different warmup
        HyperparamConfig(2e-5, 16, 0.0, 0.01, 10),
        HyperparamConfig(2e-5, 16, 0.2, 0.01, 10),
        # Different weight decay
        HyperparamConfig(2e-5, 16, 0.1, 0.0, 10),
        HyperparamConfig(2e-5, 16, 0.1, 0.1, 10),
    ]
    
    print(f"Testing {len(configs)} configurations")
    print(f"Estimated time: ~{len(configs) * 5} minutes\n")
    
    results = []
    for i, config in enumerate(configs, 1):
        print(f"\n{'#'*80}")
        print(f"Configuration {i}/{len(configs)}")
        print(f"{'#'*80}")
        
        try:
            result = train_with_hyperparams(config, model_name)
            results.append(result)
            
            print(f"\n✓ Validation F1: {result['val_f1']:.4f}")
            
        except Exception as e:
            print(f"\n✗ Failed: {e}")
            continue
    
    return results


def analyze_results(results: List[Dict]):
    """Analyze tuning results and find best configuration."""
    
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING RESULTS")
    print("="*80 + "\n")
    
    if not results:
        print("❌ No results to analyze")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('val_f1', ascending=False)
    
    # Display top 10 configurations
    print("TOP 10 CONFIGURATIONS")
    print("="*80)
    display_cols = ['learning_rate', 'batch_size', 'warmup_ratio', 'weight_decay', 'val_f1', 'val_precision', 'val_recall']
    print(df[display_cols].head(10).to_string(index=False))
    
    # Best configuration
    best = df.iloc[0]
    print(f"\n{'='*80}")
    print("BEST CONFIGURATION")
    print("="*80)
    print(f"Learning Rate: {best['learning_rate']}")
    print(f"Batch Size: {int(best['batch_size'])}")
    print(f"Warmup Ratio: {best['warmup_ratio']}")
    print(f"Weight Decay: {best['weight_decay']}")
    print(f"\nValidation Metrics:")
    print(f"  F1 Score:  {best['val_f1']:.4f}")
    print(f"  Precision: {best['val_precision']:.4f}")
    print(f"  Recall:    {best['val_recall']:.4f}")
    print(f"  Loss:      {best['val_loss']:.4f}")
    
    # Compare to baseline
    baseline_f1 = 0.8779  # Current BioBERT result
    improvement = (best['val_f1'] - baseline_f1) * 100
    
    print(f"\n{'='*80}")
    print("COMPARISON TO BASELINE")
    print("="*80)
    print(f"Baseline F1 (test):  0.8779")
    print(f"Best tuned F1 (val): {best['val_f1']:.4f}")
    print(f"Improvement:         {improvement:+.2f}%")
    
    if improvement > 0:
        print(f"\n✓ IMPROVEMENT FOUND!")
        print(f"  Recommendation: Retrain with best config on full dataset")
    else:
        print(f"\n⚠ No improvement over baseline")
        print(f"  Current hyperparameters are near-optimal")
    
    # Save results
    results_path = 'results/hyperparameter_tuning.csv'
    df.to_csv(results_path, index=False)
    print(f"\n✓ All results saved to: {results_path}")
    
    # Save best config
    best_config_path = 'results/best_hyperparameters.json'
    best_config = {
        'learning_rate': float(best['learning_rate']),
        'batch_size': int(best['batch_size']),
        'warmup_ratio': float(best['warmup_ratio']),
        'weight_decay': float(best['weight_decay']),
        'num_epochs': int(best['num_epochs']),
        'val_f1': float(best['val_f1']),
        'val_precision': float(best['val_precision']),
        'val_recall': float(best['val_recall']),
    }
    
    with open(best_config_path, 'w') as f:
        json.dump(best_config, f, indent=2)
    
    print(f"✓ Best config saved to: {best_config_path}")
    
    print(f"\n{'='*80}")
    print("✅ HYPERPARAMETER TUNING COMPLETE")
    print("="*80)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick',
                        help='Quick search (10 configs) or full grid search')
    args = parser.parse_args()
    
    if args.mode == 'quick':
        results = quick_search()
    else:
        results = grid_search()
    
    if results:
        analyze_results(results)
    else:
        print("❌ No results obtained")


if __name__ == "__main__":
    main()
