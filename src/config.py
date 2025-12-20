from dataclasses import dataclass
from typing import List


@dataclass
class ExperimentConfig:
    # Models to compare
    model_names: List[str] = (
        "bert-base-uncased",
        "roberta-base",
        "microsoft/deberta-base",
    )

    # Dataset
    dataset_name: str = "bigbio/bc5cdr"
    dataset_config: str = "bc5cdr_bigbio_kb"  # Knowledge base schema with entities

    # Hyperparameters (identical across models)
    seed: int = 42
    num_train_epochs: int = 30
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_length: int = 256
    label_all_tokens: bool = False

    # Paths
    output_dir: str = "models"
    results_csv: str = "results/metrics.csv"

    # Trainer/runtime
    fp16: bool = True  # will be overridden to False if no CUDA
    logging_steps: int = 50
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    early_stopping_patience: int = 5  # stop if no improvement for 5 epochs
    report_to: List[str] = ()  # disable W&B etc by default
