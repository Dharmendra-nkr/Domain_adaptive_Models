# Biomedical NER with DAPT-ready Pipeline (BC5CDR)

This project fine-tunes multiple pretrained language models on the BC5CDR dataset (BigBio `bigbio_ner` schema) for biomedical Named Entity Recognition (NER). It uses the same dataset and identical hyperparameters to compare models.

## Requirements
- Python 3.11 (Conda environment recommended)
- PyTorch with CUDA (installed)
- Packages: `transformers`, `datasets`, `seqeval`, `accelerate`, `pandas`, `scikit-learn`

## Project Layout
- `src/config.py` — configuration and hyperparameters
- `src/data_utils.py` — dataset loading, tokenization, and label alignment
- `src/train.py` — single-model training + evaluation with `Trainer`
- `src/evaluate.py` — load a saved model directory and evaluate on test
- `src/run_experiments.py` — loop over multiple models, save CSV results
- `models/` — checkpoints per model
- `results/metrics.csv` — comparison table

## Quickstart
Activate your environment and run:

```powershell
# Train and evaluate all default models
python -m src.run_experiments
```

Or train a single model:
```powershell
python -m src.train
```

## Example: Customizing Models / Hyperparameters
Edit `src/config.py`:
- `model_names`: list of HF model IDs to compare
- `num_train_epochs`, `learning_rate`, `per_device_train_batch_size`, etc.

## Notes
- Mixed precision (fp16) is enabled automatically if a CUDA GPU is available.
- The dataset is loaded as `load_dataset("bigbio/bc5cdr", "bigbio_ner")` (provides `tokens` and `ner_tags`).
- Metrics use `seqeval` (Precision, Recall, F1) on the test split.
