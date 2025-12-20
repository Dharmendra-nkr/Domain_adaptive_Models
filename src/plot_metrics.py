"""
Generate comparison plots for BC5CDR NER model metrics.
Reads metrics.csv and produces bar charts for precision, recall, F1.
"""
import csv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import os

# Read metrics CSV
csv_path = "results/metrics.csv"
models = []
precision_scores = []
recall_scores = []
f1_scores = []

with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        models.append(row['model_name'])
        precision_scores.append(float(row['precision']))
        recall_scores.append(float(row['recall']))
        f1_scores.append(float(row['f1']))

# Create output directory
os.makedirs("results/plots", exist_ok=True)

# Define colors
all_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
colors = [all_colors[i % len(all_colors)] for i in range(len(models))]

# Plot 1: All metrics comparison
fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(models))
width = 0.25

color_p = '#1f77b4'
color_r = '#ff7f0e'
color_f1 = '#2ca02c'

bars1 = ax.bar([i - width for i in x], precision_scores, width, label='Precision', color=color_p, alpha=0.8)
bars2 = ax.bar(x, recall_scores, width, label='Recall', color=color_r, alpha=0.8)
bars3 = ax.bar([i + width for i in x], f1_scores, width, label='F1', color=color_f1, alpha=0.8)

ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('BC5CDR NER Model Comparison - All Metrics', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim([0.8, 0.9])
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('results/plots/all_metrics.png', dpi=150, bbox_inches='tight')
print("✓ Saved: results/plots/all_metrics.png")
plt.close()

# Plot 2: F1 Score only
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(models, f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax.set_title('BC5CDR NER - F1 Score Comparison', fontsize=14, fontweight='bold')
ax.set_ylim([0.82, 0.88])
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar, score in zip(bars, f1_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('results/plots/f1_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: results/plots/f1_comparison.png")
plt.close()

# Plot 3: Precision vs Recall
fig, ax = plt.subplots(figsize=(8, 5))
x = range(len(models))
width = 0.35

bars1 = ax.bar([i - width/2 for i in x], precision_scores, width, label='Precision', color='#2ca02c', alpha=0.8)
bars2 = ax.bar([i + width/2 for i in x], recall_scores, width, label='Recall', color='#d62728', alpha=0.8)

ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('BC5CDR NER - Precision vs Recall', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim([0.82, 0.91])
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('results/plots/precision_recall.png', dpi=150, bbox_inches='tight')
print("✓ Saved: results/plots/precision_recall.png")
plt.close()

# Summary table
print("\n" + "="*60)
print("BC5CDR NER Model Evaluation Results")
print("="*60)
print(f"{'Model':<25} {'Precision':<12} {'Recall':<12} {'F1':<12}")
print("-"*60)
for model, p, r, f1 in zip(models, precision_scores, recall_scores, f1_scores):
    print(f"{model:<25} {p:<12.4f} {r:<12.4f} {f1:<12.4f}")
print("="*60)

print("\n✓ All plots saved to results/plots/")
