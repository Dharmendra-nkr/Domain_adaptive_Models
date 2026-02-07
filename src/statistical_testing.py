"""
Run statistical significance testing by training models multiple times with different seeds.
This provides mean ± std for all metrics, required for publication.
"""
import argparse
import csv
import json
import os
from typing import List

import numpy as np
import pandas as pd
from scipy import stats

from .config import ExperimentConfig
from .train import train_single_model


def run_multiple_seeds(
    model_name: str,
    seeds: List[int],
    cfg: ExperimentConfig
) -> dict:
    """
    Train a model multiple times with different seeds.
    
    Args:
        model_name: Model to train
        seeds: List of random seeds
        cfg: Experiment configuration
    
    Returns:
        Dictionary with mean, std, and all runs
    """
    results = []
    
    for seed_idx, seed in enumerate(seeds, 1):
        print(f"\n{'='*70}")
        print(f"Training {model_name} - Run {seed_idx}/{len(seeds)} (seed={seed})")
        print(f"{'='*70}\n")
        
        # Update config with new seed
        cfg.seed = seed
        
        try:
            result = train_single_model(model_name, cfg)
            result['seed'] = seed
            results.append(result)
            
            print(f"\n✓ Run {seed_idx} completed: F1={result['f1']:.4f}")
            
        except Exception as e:
            print(f"\n✗ Run {seed_idx} failed: {e}")
            continue
    
    if not results:
        raise RuntimeError(f"All runs failed for {model_name}")
    
    # Calculate statistics
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    f1s = [r['f1'] for r in results]
    
    stats_dict = {
        'model_name': model_name,
        'num_runs': len(results),
        'precision_mean': np.mean(precisions),
        'precision_std': np.std(precisions),
        'recall_mean': np.mean(recalls),
        'recall_std': np.std(recalls),
        'f1_mean': np.mean(f1s),
        'f1_std': np.std(f1s),
        'all_runs': results
    }
    
    return stats_dict


def compare_models_significance(results_df: pd.DataFrame, baseline_model: str = "bert-base-uncased"):
    """
    Perform statistical significance tests between models.
    
    Args:
        results_df: DataFrame with model results
        baseline_model: Model to compare against
    
    Returns:
        DataFrame with p-values
    """
    print(f"\n{'='*70}")
    print("Statistical Significance Testing (t-test)")
    print(f"Baseline: {baseline_model}")
    print(f"{'='*70}\n")
    
    # Load all individual runs
    all_runs = {}
    for _, row in results_df.iterrows():
        model = row['model_name']
        runs = eval(row['all_runs']) if isinstance(row['all_runs'], str) else row['all_runs']
        all_runs[model] = [r['f1'] for r in runs]
    
    baseline_f1s = all_runs.get(baseline_model)
    if not baseline_f1s:
        print(f"Warning: Baseline model {baseline_model} not found")
        return None
    
    significance_results = []
    
    for model, f1s in all_runs.items():
        if model == baseline_model:
            continue
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(f1s, baseline_f1s)
        
        # Determine significance level
        if p_value < 0.001:
            sig = "***"
        elif p_value < 0.01:
            sig = "**"
        elif p_value < 0.05:
            sig = "*"
        else:
            sig = "ns"
        
        significance_results.append({
            'model': model,
            't_statistic': t_stat,
            'p_value': p_value,
            'significance': sig,
            'mean_diff': np.mean(f1s) - np.mean(baseline_f1s)
        })
        
        print(f"{model:50s} p={p_value:.4f} {sig:3s} (Δ={np.mean(f1s) - np.mean(baseline_f1s):+.4f})")
    
    print(f"\n* p<0.05, ** p<0.01, *** p<0.001, ns=not significant")
    
    return pd.DataFrame(significance_results)


def main():
    parser = argparse.ArgumentParser(description="Statistical significance testing")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["bert-base-uncased", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"],
        help="Models to test"
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
        help="Random seeds to use"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/statistical_significance.csv",
        help="Output CSV path"
    )
    
    args = parser.parse_args()
    
    cfg = ExperimentConfig()
    all_results = []
    
    print(f"\n{'='*70}")
    print("STATISTICAL SIGNIFICANCE TESTING")
    print(f"{'='*70}")
    print(f"Models: {len(args.models)}")
    print(f"Seeds per model: {len(args.seeds)}")
    print(f"Total experiments: {len(args.models) * len(args.seeds)}")
    print(f"{'='*70}\n")
    
    for model in args.models:
        stats_dict = run_multiple_seeds(model, args.seeds, cfg)
        all_results.append(stats_dict)
        
        print(f"\n{model}:")
        print(f"  Precision: {stats_dict['precision_mean']:.4f} ± {stats_dict['precision_std']:.4f}")
        print(f"  Recall:    {stats_dict['recall_mean']:.4f} ± {stats_dict['recall_std']:.4f}")
        print(f"  F1:        {stats_dict['f1_mean']:.4f} ± {stats_dict['f1_std']:.4f}")
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    df = pd.DataFrame(all_results)
    df.to_csv(args.output, index=False)
    
    print(f"\n✓ Results saved to: {args.output}")
    
    # Significance testing
    sig_df = compare_models_significance(df)
    if sig_df is not None:
        sig_output = args.output.replace('.csv', '_significance.csv')
        sig_df.to_csv(sig_output, index=False)
        print(f"✓ Significance tests saved to: {sig_output}")
    
    print(f"\n{'='*70}")
    print("STATISTICAL TESTING COMPLETED")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
