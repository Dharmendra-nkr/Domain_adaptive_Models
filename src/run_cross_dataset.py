"""
Cross-dataset evaluation script.
Trains and evaluates models across multiple biomedical NER datasets.
"""
import argparse
import csv
import os
from typing import List, Optional

import pandas as pd

from .config import ExperimentConfig
from .dapt_config import CROSS_DATASETS, DatasetConfig
from .train import train_single_model


def run_cross_dataset_experiments(
    models: List[str],
    datasets: Optional[List[str]] = None,
    output_csv: str = "results/cross_dataset_metrics.csv",
):
    """
    Run experiments across multiple datasets.
    
    Args:
        models: List of model names to evaluate
        datasets: List of dataset names (uses all if None)
        output_csv: Path to save results CSV
    """
    # Filter datasets
    if datasets:
        dataset_configs = [d for d in CROSS_DATASETS if d.display_name in datasets]
    else:
        dataset_configs = CROSS_DATASETS
    
    print("="*70)
    print("CROSS-DATASET EVALUATION")
    print("="*70)
    print(f"Models: {len(models)}")
    for m in models:
        print(f"  - {m}")
    print(f"\nDatasets: {len(dataset_configs)}")
    for d in dataset_configs:
        print(f"  - {d.display_name}: {d.description}")
    print("="*70 + "\n")
    
    # Collect results
    all_results = []
    total_experiments = len(models) * len(dataset_configs)
    current_experiment = 0
    
    for dataset_cfg in dataset_configs:
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_cfg.display_name}")
        print(f"{'='*70}\n")
        
        for model_name in models:
            current_experiment += 1
            print(f"\n[{current_experiment}/{total_experiments}] Training {model_name} on {dataset_cfg.display_name}")
            print("-"*70)
            
            try:
                # Create experiment config for this dataset
                exp_cfg = ExperimentConfig(
                    dataset_name=dataset_cfg.name,
                    dataset_config=dataset_cfg.config,
                )
                
                # Train and evaluate
                result = train_single_model(model_name, exp_cfg)
                
                # Add dataset info
                result["dataset"] = dataset_cfg.display_name
                result["dataset_name"] = dataset_cfg.name
                result["dataset_config"] = dataset_cfg.config
                
                all_results.append(result)
                
                print(f"✓ Completed: F1={result['f1']:.4f}, P={result['precision']:.4f}, R={result['recall']:.4f}")
                
            except Exception as e:
                print(f"✗ Failed: {e}")
                # Log failure but continue
                all_results.append({
                    "model_name": model_name,
                    "dataset": dataset_cfg.display_name,
                    "dataset_name": dataset_cfg.name,
                    "dataset_config": dataset_cfg.config,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "output_dir": "",
                    "error": str(e)
                })
    
    # Save results
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    fieldnames = ["dataset", "model_name", "precision", "recall", "f1", "output_dir", "dataset_name", "dataset_config"]
    if any("error" in r for r in all_results):
        fieldnames.append("error")
    
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)
    
    print(f"\n{'='*70}")
    print("CROSS-DATASET EVALUATION COMPLETED")
    print(f"{'='*70}")
    print(f"Results saved to: {output_csv}")
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {sum(1 for r in all_results if r.get('f1', 0) > 0)}")
    print(f"Failed: {sum(1 for r in all_results if r.get('f1', 0) == 0)}")
    
    # Print summary table
    print("\nResults Summary:")
    df = pd.DataFrame(all_results)
    if len(df) > 0:
        summary = df.pivot_table(
            values='f1',
            index='model_name',
            columns='dataset',
            aggfunc='mean'
        )
        print(summary.to_string())
    
    print("="*70 + "\n")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cross-dataset NER experiments")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="List of model names to evaluate"
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="List of dataset names (default: all available)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/cross_dataset_metrics.csv",
        help="Output CSV path"
    )
    
    args = parser.parse_args()
    
    run_cross_dataset_experiments(
        models=args.models,
        datasets=args.datasets,
        output_csv=args.output
    )
