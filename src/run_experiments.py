"""
Run end-to-end experiments:
- Trains and evaluates multiple pretrained models on BC5CDR (same hyperparams)
- Saves per-model metrics to CSV for comparison
- Supports appending new models to existing results
"""
import csv
import os
import argparse
from typing import List

from .config import ExperimentConfig
from .train import train_single_model


def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def load_existing_results(csv_path: str) -> dict:
    """Load existing results from CSV into a dict keyed by model_name"""
    results_dict = {}
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results_dict[row["model_name"]] = row
    return results_dict


def run_all(cfg: ExperimentConfig, models: List[str] | None = None, append: bool = True):
    models = models or list(cfg.model_names)
    ensure_dir(cfg.results_csv)

    # Load existing results if appending
    if append:
        existing_results = load_existing_results(cfg.results_csv)
        models_to_train = [m for m in models if m not in existing_results]
        if models_to_train:
            print(f"Found {len(existing_results)} existing results. Training {len(models_to_train)} new models...")
        else:
            print(f"All {len(models)} models already trained. Skipping training.")
            return
    else:
        existing_results = {}
        models_to_train = models

    # Train new models
    new_results = []
    for m in models_to_train:
        print(f"\n===== Training {m} =====")
        res = train_single_model(m, cfg)
        new_results.append(res)
        print(f"Result: {res}")

    # Combine existing + new results
    all_results = list(existing_results.values()) + new_results

    # Write combined CSV
    with open(cfg.results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model_name", "precision", "recall", "f1", "output_dir"])
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)

    print(f"Saved {len(all_results)} results to {cfg.results_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BC5CDR experiments across selected models")
    parser.add_argument(
        "--models",
        nargs="*",
        help="Optional list of model names to run (default runs all from config)",
    )
    parser.add_argument(
        "--no-append",
        action="store_true",
        help="Overwrite CSV instead of appending (default: append mode)",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig()
    run_all(cfg, models=args.models, append=not args.no_append)
