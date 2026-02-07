"""
Configuration for Domain-Adaptive Pre-Training (DAPT).
Defines hyperparameters for continuing pre-training on biomedical corpus.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class DAPTConfig:
    """Configuration for DAPT pre-training."""
    
    # Corpus settings
    corpus_name: str = "pubmed"  # or "pubmed_abstracts"
    corpus_subset: Optional[str] = "train[:100000]"  # Use subset for faster training
    
    # Model to continue pre-training
    base_model: str = "bert-base-uncased"
    output_dir: str = "models/bert-base-dapt"
    
    # MLM settings
    mlm_probability: float = 0.15  # Standard BERT masking rate
    
    # Training hyperparameters
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 16  # Adjust based on GPU memory
    per_device_eval_batch_size: int = 32
    learning_rate: float = 5e-5  # Higher than fine-tuning, lower than pre-training from scratch
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 512  # Full BERT context
    
    # Performance settings
    fp16: bool = True  # Mixed precision for faster training
    gradient_accumulation_steps: int = 2  # Effective batch size = 16 * 2 = 32
    dataloader_num_workers: int = 4
    
    # Logging and checkpointing
    logging_steps: int = 100
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    save_total_limit: int = 2  # Keep last 2 checkpoints
    
    # Data processing
    preprocessing_num_workers: int = 4
    overwrite_cache: bool = False
    
    # Validation
    validation_split_percentage: int = 5  # 5% for validation
    
    # Seed for reproducibility
    seed: int = 42


@dataclass
class DatasetConfig:
    """Configuration for a single NER dataset."""
    name: str
    config: str
    display_name: str
    description: str = ""


# Multiple datasets for cross-dataset validation
CROSS_DATASETS = [
    DatasetConfig(
        name="bigbio/bc5cdr",
        config="bc5cdr_bigbio_kb",
        display_name="BC5CDR",
        description="Chemical and Disease entities from PubMed abstracts"
    ),
    DatasetConfig(
        name="bigbio/ncbi_disease",
        config="ncbi_disease_bigbio_kb",
        display_name="NCBI-Disease",
        description="Disease mentions from PubMed abstracts"
    ),
    DatasetConfig(
        name="bigbio/bc4chemd",
        config="bc4chemd_bigbio_kb",
        display_name="BC4CHEMD",
        description="Chemical entities from PubMed abstracts"
    ),
]
