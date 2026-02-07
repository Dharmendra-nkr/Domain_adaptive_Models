"""
Visualization for cross-dataset evaluation results.
Creates heatmaps, comparison plots, and ranking analysis.
"""
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_cross_dataset_heatmap(
    csv_path: str = "results/cross_dataset_metrics.csv",
    output_path: str = "results/plots/cross_dataset_heatmap.png",
    metric: str = "f1"
):
    """
    Create heatmap showing model performance across datasets.
    
    Args:
        csv_path: Path to cross-dataset results CSV
        output_path: Path to save plot
        metric: Metric to visualize (f1, precision, recall)
    """
    df = pd.read_csv(csv_path)
    
    # Pivot table: models as rows, datasets as columns
    pivot = df.pivot_table(
        values=metric,
        index='model_name',
        columns='dataset',
        aggfunc='mean'
    )
    
    # Sort by average performance
    pivot['average'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('average', ascending=False)
    pivot = pivot.drop('average', axis=1)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0.7,
        vmax=0.95,
        cbar_kws={'label': metric.upper()},
        linewidths=0.5
    )
    
    plt.title(f'Cross-Dataset Model Performance ({metric.upper()})', fontsize=16, fontweight='bold')
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to: {output_path}")
    plt.close()


def plot_model_ranking_consistency(
    csv_path: str = "results/cross_dataset_metrics.csv",
    output_path: str = "results/plots/model_ranking_consistency.png"
):
    """
    Visualize how model rankings change across datasets.
    
    Args:
        csv_path: Path to cross-dataset results CSV
        output_path: Path to save plot
    """
    df = pd.read_csv(csv_path)
    
    datasets = df['dataset'].unique()
    
    fig, axes = plt.subplots(1, len(datasets), figsize=(6*len(datasets), 6), sharey=True)
    
    if len(datasets) == 1:
        axes = [axes]
    
    for idx, dataset in enumerate(datasets):
        dataset_df = df[df['dataset'] == dataset].sort_values('f1', ascending=True)
        
        ax = axes[idx]
        colors = plt.cm.viridis(np.linspace(0, 1, len(dataset_df)))
        
        ax.barh(dataset_df['model_name'], dataset_df['f1'], color=colors)
        ax.set_xlabel('F1 Score', fontsize=12)
        ax.set_title(dataset, fontsize=14, fontweight='bold')
        ax.set_xlim(0.7, 0.95)
        
        # Add value labels
        for i, (name, f1) in enumerate(zip(dataset_df['model_name'], dataset_df['f1'])):
            ax.text(f1 + 0.005, i, f'{f1:.3f}', va='center', fontsize=10)
    
    axes[0].set_ylabel('Model', fontsize=12)
    
    plt.suptitle('Model Rankings Across Datasets', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved ranking plot to: {output_path}")
    plt.close()


def plot_dapt_comparison(
    csv_path: str = "results/metrics.csv",
    output_path: str = "results/plots/dapt_comparison.png",
    dapt_model_name: str = "models/bert-base-dapt"
):
    """
    Create visualization comparing BERT, BERT+DAPT, and BioBERT.
    
    Args:
        csv_path: Path to results CSV
        output_path: Path to save plot
        dapt_model_name: Name of DAPT model in CSV
    """
    df = pd.read_csv(csv_path)
    
    # Filter relevant models
    models_of_interest = [
        'bert-base-uncased',
        dapt_model_name,
        'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
    ]
    
    df_filtered = df[df['model_name'].isin(models_of_interest)]
    
    if len(df_filtered) == 0:
        print(f"Warning: No DAPT results found. Expected model name: {dapt_model_name}")
        return
    
    # Rename for clarity
    name_mapping = {
        'bert-base-uncased': 'BERT (baseline)',
        dapt_model_name: 'BERT + DAPT',
        'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext': 'BioBERT'
    }
    df_filtered['display_name'] = df_filtered['model_name'].map(name_mapping)
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 7))
    
    metrics = ['precision', 'recall', 'f1']
    x = np.arange(len(df_filtered))
    width = 0.25
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for i, metric in enumerate(metrics):
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset,
            df_filtered[metric],
            width,
            label=metric.capitalize(),
            color=colors[i],
            alpha=0.8
        )
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.005,
                f'{height:.3f}',
                ha='center',
                va='bottom',
                fontsize=10
            )
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('DAPT Effectiveness: BERT → BERT+DAPT → BioBERT', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_filtered['display_name'], fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0.7, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved DAPT comparison to: {output_path}")
    plt.close()


def generate_all_cross_dataset_plots(
    csv_path: str = "results/cross_dataset_metrics.csv",
    output_dir: str = "results/plots"
):
    """Generate all cross-dataset visualizations."""
    print("Generating cross-dataset visualizations...")
    
    if not os.path.exists(csv_path):
        print(f"Error: Results file not found: {csv_path}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Heatmap
    plot_cross_dataset_heatmap(
        csv_path=csv_path,
        output_path=os.path.join(output_dir, "cross_dataset_heatmap.png")
    )
    
    # Ranking consistency
    plot_model_ranking_consistency(
        csv_path=csv_path,
        output_path=os.path.join(output_dir, "model_ranking_consistency.png")
    )
    
    print("All cross-dataset plots generated!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate cross-dataset visualizations")
    parser.add_argument(
        "--csv",
        type=str,
        default="results/cross_dataset_metrics.csv",
        help="Path to cross-dataset results CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/plots",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--dapt",
        action="store_true",
        help="Also generate DAPT comparison plot"
    )
    
    args = parser.parse_args()
    
    generate_all_cross_dataset_plots(
        csv_path=args.csv,
        output_dir=args.output_dir
    )
    
    if args.dapt:
        plot_dapt_comparison(
            csv_path="results/metrics.csv",
            output_path=os.path.join(args.output_dir, "dapt_comparison.png")
        )
