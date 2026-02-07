"""
Utility functions for DAPT (Domain-Adaptive Pre-Training).
Handles corpus loading, preprocessing, and dataset preparation.
"""
import re
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset


def load_pubmed_corpus(
    corpus_name: str = "pubmed",
    subset: Optional[str] = None,
    validation_split: float = 0.05,
) -> DatasetDict:
    """
    Load PubMed corpus for domain-adaptive pre-training.
    
    Args:
        corpus_name: Name of the corpus dataset
        subset: Optional subset specification (e.g., "train[:100000]")
        validation_split: Fraction of data to use for validation
    
    Returns:
        DatasetDict with 'train' and 'validation' splits
    """
    print(f"Loading biomedical corpus...")
    
    # Try multiple sources in order of preference
    sources_to_try = [
        ("ccdv/pubmed-summarization", "document", "PubMed Summarization"),
        ("scientific_papers", "pubmed", "Scientific Papers PubMed"),
        ("bigbio/pubmed", "pubmed_bigbio_kb", "BigBio PubMed"),
    ]
    
    for dataset_name, config_name, display_name in sources_to_try:
        try:
            print(f"  Trying {display_name}...")
            
            if subset:
                dataset = load_dataset(dataset_name, config_name, split=subset, trust_remote_code=True)
            else:
                dataset = load_dataset(dataset_name, config_name, split="train[:100000]", trust_remote_code=True)
            
            # If dataset is a single split, create train/val split
            if isinstance(dataset, Dataset):
                split_dataset = dataset.train_test_split(
                    test_size=validation_split,
                    seed=42
                )
                print(f"  ✓ Successfully loaded {display_name}")
                return DatasetDict({
                    "train": split_dataset["train"],
                    "validation": split_dataset["test"]
                })
            else:
                print(f"  ✓ Successfully loaded {display_name}")
                return dataset
                
        except Exception as e:
            print(f"  ✗ Failed to load {display_name}: {str(e)[:100]}")
            continue
    
    # If all sources fail, raise error
    raise RuntimeError(
        "Failed to load any biomedical corpus. Tried multiple sources.\n"
        "Please check your internet connection and ensure 'datasets' library is up to date.\n"
        "You can also manually download a biomedical text corpus and modify this function."
    )



def clean_biomedical_text(text: str) -> str:
    """
    Clean and normalize biomedical text.
    
    Args:
        text: Raw text string
    
    Returns:
        Cleaned text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that might interfere with tokenization
    # Keep alphanumeric, spaces, and common punctuation
    text = re.sub(r'[^\w\s.,;:!?()\-\'/]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def extract_text_field(example: Dict, text_fields: List[str] = None) -> Dict:
    """
    Extract and clean text from dataset example.
    Handles different dataset schemas (abstract, text, article, etc.)
    
    Args:
        example: Dataset example dictionary
        text_fields: List of possible text field names to try
    
    Returns:
        Dictionary with 'text' field
    """
    if text_fields is None:
        text_fields = ['abstract', 'text', 'article', 'content', 'passage']
    
    text = ""
    for field in text_fields:
        if field in example and example[field]:
            text = example[field]
            break
    
    # If text is a list (some datasets have multiple passages), join them
    if isinstance(text, list):
        text = " ".join(str(t) for t in text if t)
    
    # Clean the text
    text = clean_biomedical_text(str(text))
    
    return {"text": text}


def prepare_mlm_dataset(
    dataset: DatasetDict,
    tokenizer,
    max_length: int = 512,
    num_workers: int = 4,
) -> DatasetDict:
    """
    Prepare dataset for Masked Language Modeling.
    
    Args:
        dataset: Raw dataset with text
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        num_workers: Number of preprocessing workers
    
    Returns:
        Tokenized dataset ready for MLM training
    """
    print("Preprocessing dataset for MLM...")
    
    # Extract and clean text
    dataset = dataset.map(
        extract_text_field,
        num_proc=num_workers,
        desc="Extracting text fields"
    )
    
    # Filter out empty texts
    dataset = dataset.filter(
        lambda x: x["text"] and len(x["text"]) > 10,
        num_proc=num_workers,
        desc="Filtering empty texts"
    )
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,  # Dynamic padding in data collator
            return_special_tokens_mask=True,
        )
    
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_workers,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing texts"
    )
    
    print(f"Prepared {len(tokenized['train'])} training examples")
    print(f"Prepared {len(tokenized['validation'])} validation examples")
    
    return tokenized


def estimate_training_time(
    num_examples: int,
    batch_size: int,
    num_epochs: int,
    seconds_per_batch: float = 0.5
) -> Tuple[float, str]:
    """
    Estimate training time for DAPT.
    
    Args:
        num_examples: Number of training examples
        batch_size: Effective batch size
        num_epochs: Number of training epochs
        seconds_per_batch: Estimated seconds per batch (depends on GPU)
    
    Returns:
        Tuple of (total_seconds, human_readable_string)
    """
    num_batches = (num_examples // batch_size) * num_epochs
    total_seconds = num_batches * seconds_per_batch
    
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    
    time_str = f"{int(hours)}h {int(minutes)}m"
    
    return total_seconds, time_str


def print_dataset_stats(dataset: DatasetDict):
    """Print statistics about the dataset."""
    print("\n" + "="*50)
    print("Dataset Statistics")
    print("="*50)
    
    for split_name, split_data in dataset.items():
        print(f"\n{split_name.upper()} Split:")
        print(f"  Examples: {len(split_data):,}")
        
        if "text" in split_data.column_names:
            # Calculate average text length
            sample_size = min(1000, len(split_data))
            sample = split_data.select(range(sample_size))
            avg_length = sum(len(x["text"]) for x in sample) / sample_size
            print(f"  Avg text length: {int(avg_length)} characters")
    
    print("="*50 + "\n")
