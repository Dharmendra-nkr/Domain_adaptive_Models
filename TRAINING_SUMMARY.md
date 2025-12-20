# BC5CDR Biomedical NER - Training Summary

## Overview
Training of three pre-trained language models (PLMs) on BC5CDR biomedical named entity recognition task using Hugging Face Transformers.

## Results

### ✅ Completed Models

#### 1. BERT (bert-base-uncased)
- **Status**: ✅ Successfully trained (20 epochs)
- **Precision**: 0.8243
- **Recall**: 0.8690
- **F1 Score**: 0.8460
- **Training Time**: ~370 seconds
- **Model Path**: `models/bert-base-uncased/`

#### 2. RoBERTa (roberta-base)
- **Status**: ✅ Successfully trained (25 epochs, early stopping triggered)
- **Precision**: 0.8492
- **Recall**: 0.8931
- **F1 Score**: 0.8706 ⭐ **Best Performance**
- **Training Time**: ~697 seconds
- **Model Path**: `models/roberta-base/`

**RoBERTa outperformed BERT with +2.46% F1 improvement** (0.8706 vs 0.8460)

### ❌ Pending Models

#### 3. DeBERTa (microsoft/deberta-base)
- **Status**: ⏳ Not yet trained
- **Issue**: Environment stability issues causing import/runtime hangs
- **Details**: 
  - DeBERTa requires FP32 (full precision) instead of FP16 due to numerical overflow bug
  - Fix was applied but training attempts blocked by:
    - HuggingFace Hub API timeout/hang issues during metadata queries
    - Import chain deadlock in transformer/datasets libraries
    - Subprocess/conda environment instability

## Configuration

- **Dataset**: BC5CDR (BigBio KB schema)
  - Train: 500 documents
  - Validation: 500 documents
  - Test: 500 documents
  
- **Hyperparameters** (identical across models):
  - Learning Rate: 2e-5
  - Max Epochs: 30
  - Early Stopping: 5 patience
  - Batch Size (train/eval): 16/32
  - Mixed Precision: FP16 (except DeBERTa → FP32)
  - Save Strategy: Best model only (save_total_limit=1)

- **Task**: Token Classification (NER)
  - Labels: B-Chemical, I-Chemical, B-Disease, I-Disease, O

## Comparison Plots

Generated visualizations saved to `results/plots/`:
- ✅ `all_metrics.png` - Side-by-side precision/recall/F1 comparison
- ✅ `f1_comparison.png` - F1 score bar chart
- ✅ `precision_recall.png` - Precision vs recall comparison

## Metrics CSV

Results saved to `results/metrics.csv`:
```
model_name,precision,recall,f1,output_dir
bert-base-uncased,0.8243169398907104,0.8689516129032258,0.8460459899046551,models/bert-base-uncased
roberta-base,0.8492435399651894,0.8931286961419318,0.8706334500034314,models/roberta-base
```

## Next Steps for DeBERTa

1. **Restart Environment**: Fresh conda env or Python kernel to clear import state
2. **Pre-download Models**: Manual download of DeBERTa weights before training
3. **Alternative**: Use HF Accelerate library or lower-level PyTorch training loop
4. **Backup**: Can compare BERT/RoBERTa only if DeBERTa cannot be resolved

## Key Takeaways

- **RoBERTa is the best performer** for this BC5CDR task (F1: 0.8706)
- Both BERT and RoBERTa show strong performance (>0.84 F1)
- RoBERTa's superior recall (0.8931 vs 0.8690) is key advantage
- DeBERTa blocked by environment/system issues, not model architecture

## Files Generated

```
results/
  ├── metrics.csv                    # Results table (BERT, RoBERTa)
  └── plots/
      ├── all_metrics.png
      ├── f1_comparison.png
      └── precision_recall.png

models/
  ├── bert-base-uncased/
  │   ├── eval_metrics.json
  │   └── [checkpoint files]
  ├── roberta-base/
  │   ├── eval_metrics.json
  │   └── [checkpoint files]
  └── microsoft--deberta-base/       # (empty - not trained)

src/
  ├── train.py                       # Training script with fixes applied
  ├── run_experiments.py             # (CLI args added)
  ├── plot_metrics.py                # Plotting script
  ├── data_utils.py                  # Data loading & tokenization
  └── config.py                      # Configuration
```

## Recommendations

1. ✅ Use **RoBERTa** for production deployment (best F1 score)
2. 🔄 Retry DeBERTa training once environment is stable
3. 📊 All plots and metrics are reproducible via `python src/plot_metrics.py`
4. 💾 Model checkpoints saved with automatic cleanup (save_total_limit=1)

---

**Status as of**: December 18, 2025  
**Models Completed**: 2/3  
**Overall Success Rate**: 67%
