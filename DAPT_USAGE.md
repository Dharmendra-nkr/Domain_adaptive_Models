# DAPT & Cross-Dataset Validation - Quick Start Guide

## Overview
This guide explains how to run Domain-Adaptive Pre-Training (DAPT) and cross-dataset validation experiments.

---

## Part 1: DAPT (Domain-Adaptive Pre-Training)

### What is DAPT?
Continue pre-training BERT on biomedical text (PubMed) to adapt it to the medical domain, then fine-tune on BC5CDR.

### Quick Start

#### Option 1: Default Settings (Recommended)
```powershell
# Run DAPT with 100K PubMed abstracts, 2 epochs
python -m src.dapt_pretraining
```

#### Option 2: Custom Settings
```powershell
# Larger corpus (500K abstracts)
python -m src.dapt_pretraining --subset "train[:500000]" --epochs 2

# Smaller corpus for testing (10K abstracts)
python -m src.dapt_pretraining --subset "train[:10000]" --epochs 1

# Different base model
python -m src.dapt_pretraining --base-model roberta-base --output-dir models/roberta-base-dapt
```

### Expected Output
- **Model checkpoint**: `models/bert-base-dapt/`
- **Training time**: 2-6 hours (depends on corpus size and GPU)
- **Validation loss**: Should decrease during training

### Next: Fine-tune DAPT Model on BC5CDR

```powershell
# Add DAPT model to experiments
python -m src.run_experiments --models bert-base-uncased models/bert-base-dapt microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
```

### Visualize DAPT Results
```powershell
python -m src.plot_cross_dataset --dapt
```

---

## Part 2: Cross-Dataset Validation

### What is Cross-Dataset Validation?
Evaluate models on multiple biomedical NER datasets to prove generalization.

### Available Datasets
1. **BC5CDR** - Chemical + Disease entities (already done)
2. **NCBI-Disease** - Disease mentions only
3. **BC4CHEMD** - Chemical entities only

### Quick Start

#### Run All Models on All Datasets
```powershell
python -m src.run_cross_dataset --models bert-base-uncased roberta-base microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext allenai/scibert_scivocab_uncased
```

#### Run Specific Models on Specific Datasets
```powershell
# Just BioBERT and SciBERT on NCBI-Disease
python -m src.run_cross_dataset --models microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext allenai/scibert_scivocab_uncased --datasets NCBI-Disease
```

#### Include DAPT Model
```powershell
python -m src.run_cross_dataset --models bert-base-uncased models/bert-base-dapt microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext --datasets BC5CDR NCBI-Disease BC4CHEMD
```

### Expected Output
- **Results CSV**: `results/cross_dataset_metrics.csv`
- **Training time**: ~30 min per model per dataset
- **Total time**: 4 models × 3 datasets × 30 min = ~6 hours

### Visualize Cross-Dataset Results
```powershell
# Generate heatmap and ranking plots
python -m src.plot_cross_dataset
```

---

## Complete Workflow Example

### Step 1: Run DAPT (one-time, ~4 hours)
```powershell
python -m src.dapt_pretraining --subset "train[:100000]" --epochs 2
```

### Step 2: Fine-tune DAPT on BC5CDR (~30 min)
```powershell
python -m src.run_experiments --models models/bert-base-dapt
```

### Step 3: Cross-Dataset Evaluation (~6 hours)
```powershell
python -m src.run_cross_dataset --models bert-base-uncased models/bert-base-dapt microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext allenai/scibert_scivocab_uncased
```

### Step 4: Generate All Visualizations
```powershell
# Cross-dataset plots
python -m src.plot_cross_dataset

# DAPT comparison
python -m src.plot_cross_dataset --dapt

# Original BC5CDR plots
python -m src.plot_metrics
```

---

## Expected Results

### DAPT Effectiveness
| Model | F1 Score | Improvement |
|-------|----------|-------------|
| BERT | 0.846 | baseline |
| BERT+DAPT | ~0.87 | +2.4% |
| BioBERT | 0.883 | +3.7% |

**Success**: DAPT should bridge 50-80% of the gap between BERT and BioBERT.

### Cross-Dataset Generalization
BioBERT and SciBERT should consistently rank in top 2 across all datasets.

---

## Troubleshooting

### Issue: "Failed to load pubmed corpus"
**Solution**: The script will automatically try alternative datasets. If all fail:
```powershell
# Use scientific_papers dataset instead
python -m src.dapt_pretraining --corpus scientific_papers
```

### Issue: Out of memory during DAPT
**Solution**: Reduce batch size
```powershell
python -m src.dapt_pretraining --batch-size 8
```

### Issue: Cross-dataset experiments taking too long
**Solution**: Run datasets sequentially or use fewer models
```powershell
# Just test on NCBI-Disease first
python -m src.run_cross_dataset --models bert-base-uncased microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext --datasets NCBI-Disease
```

---

## Files Created

### Code
- `src/dapt_config.py` - DAPT configuration
- `src/dapt_utils.py` - Corpus loading utilities
- `src/dapt_pretraining.py` - Main DAPT script
- `src/run_cross_dataset.py` - Cross-dataset evaluation
- `src/plot_cross_dataset.py` - Visualization

### Results
- `models/bert-base-dapt/` - DAPT checkpoint
- `results/cross_dataset_metrics.csv` - All results
- `results/plots/cross_dataset_heatmap.png`
- `results/plots/model_ranking_consistency.png`
- `results/plots/dapt_comparison.png`

---

## Next Steps After Completion

1. **Analyze Results**: Compare BERT vs BERT+DAPT vs BioBERT
2. **Write Paper**: Use results for publication
3. **Further Research**: Try TAPT, ensemble methods, few-shot learning

Good luck! 🚀
