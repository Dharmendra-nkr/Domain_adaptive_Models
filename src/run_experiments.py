"""
Run end-to-end experiments:
- Trains and evaluates multiple pretrained models on BC5CDR (same hyperparams)
- Saves per-model metrics to CSV for comparison
"""
import csv
import os
import argparse
from typing import List

from .config import ExperimentConfig
from .train import train_single_model


def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def run_all(cfg: ExperimentConfig, models: List[str] | None = None):
    models = models or list(cfg.model_names)
    ensure_dir(cfg.results_csv)

    results = []
    for m in models:
        print(f"\n===== Training {m} =====")
        res = train_single_model(m, cfg)
        results.append(res)
        print(f"Result: {res}")

    # Write CSV
    with open(cfg.results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model_name", "precision", "recall", "f1", "output_dir"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"Saved results to {cfg.results_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BC5CDR experiments across selected models")
    parser.add_argument(
        "--models",
        nargs="*",
        help="Optional list of model names to run (default runs all from config)",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig()
    run_all(cfg, models=args.models)
