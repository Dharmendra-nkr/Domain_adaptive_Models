# Pending Works - Progress Report

## Status Summary
**Started:** 2026-02-01 (Session Start)  
**Current Time:** Ongoing

---

## 1. ✅ DeBERTa Training - **COMPLETED**

### What Was Done
- Identified DeBERTa FP32 fix already implemented in `src/train.py`
- Verified all 7 models trained successfully:
  - BERT: F1 = 0.846
  - RoBERTa: F1 = 0.871
  - **BiomedBERT: F1 = 0.883** ⭐ (best model)
  - SciBERT: F1 = 0.877  
  - ELECTRA: F1 = 0.867
  - XLNet: F1 = 0.870
  - **DeBERTa: Trained** (fix: disabled FP16, using FP32)

### Model Directory
`models/microsoft--deberta-base/` - Checkpoint 768 saved

### Resolution
✅ No action needed - DeBERTa already trained with proper FP32 handling

---

## 2. 🔄 DAPT (Domain-Adaptive Pre-Training) - **IN PROGRESS**

### What Is Running
- Script: `run_dapt_simple.py`
- Configuration:
  - Base Model: `bert-base-uncased`
  - Corpus: PubMed (10K abstracts for quick test)
  - Epochs: 1
  - Batch Size: 8 (effective batch = 8)
  - Output: `models/bert-base-dapt/`

### Current Status
```
Starting DAPT training...
Loading tokenizer and model: bert-base-uncased
[Currently loading corpus...]
```

### Expected Completion
- **Time**: ~15-20 minutes (10K corpus, 1 epoch)
- **Output**: Pre-trained DAPT model ready for fine-tuning on BC5CDR

### Next Step
Once complete, can fine-tune DAPT-BERT on BC5CDR:
```powershell
python -m src.run_experiments --models models/bert-base-dapt
```

---

## 3. 📊 Statistical Testing - **IN PROGRESS**

### What Is Running
- Script: `src/statistical_testing`
- Models Tested (4 × 5 runs = 20 experiments):
  1. `bert-base-uncased`
  2. `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`
  3. `roberta-base`
  4. `allenai/scibert_scivocab_uncased`

### Current Progress
```
Run 1/5 for BERT: ~28% complete (256/960 steps)
Progress: Training batch iterations proceeding normally
Speed: ~2.3 iterations/sec on RTX 3050 GPU
```

### Expected Metrics Output
Publication-ready results with **mean ± standard deviation** for:
- **Precision** (across 5 runs)
- **Recall** (across 5 runs)
- **F1 Score** (across 5 runs)
- **Statistical significance** (p-values from t-tests)

### Output Files
- `results/statistical_significance.csv` - Mean/std for all models
- `results/statistical_significance_significance.csv` - P-values

### Expected Completion Time
- Current pace: ~8-10 hours for 4 models × 5 runs each
- GPU: RTX 3050 Laptop GPU (2.5GB allocated per model)
- Estimated finish: **Late evening**

---

## Environment Setup
✅ **Active Environment**: `biomedical-ner` (with PyTorch 2.5.1, CUDA 12.1)
✅ **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU  
✅ **GPU Memory**: Available and allocated
✅ **Conda**: `pytorch-gpu` removed (not needed)

---

## Summary Table

| Task | Status | Completion | Notes |
|------|--------|------------|----|
| DeBERTa | ✅ DONE | 100% | Already trained with FP32 fix |
| DAPT | 🔄 RUNNING | ~5% | Loading corpus, should start training soon |
| Statistical Testing | 🔄 RUNNING | ~5% | 256/960 steps on BERT run 1/5 |
| **Overall** | 🔄 **IN PROGRESS** | **~5%** | All GPU tasks running smoothly |

---

## Key Fixes Applied

1. **DeBERTa FP16 Issue**
   - ✅ Fixed: Automatic FP32 fallback for DeBERTa models
   - Location: `src/train.py:line 94`
   - Code: `fp16 = cfg.fp16 and torch.cuda.is_available() and "deberta" not in model_name.lower()`

2. **DAPT Implementation**
   - ✅ Created simplified runner: `run_dapt_simple.py`
   - ✅ Reduced corpus to 10K for faster validation
   - ✅ Configured for single GPU training

3. **Statistical Testing**
   - ✅ Running with robust multi-seed approach (5 runs per model)
   - ✅ Will generate publication-ready results with significance tests

---

## Next Actions (Automatic)
1. Monitor DAPT completion → Save model
2. Monitor Statistical Testing → Generate significance results
3. Optionally: Fine-tune DAPT-BERT on BC5CDR for enhanced performance

