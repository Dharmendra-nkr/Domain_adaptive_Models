"""
Dataset utilities for BC5CDR biomedical NER using Hugging Face datasets.
- Loads the BigBio BC5CDR dataset in the bc5cdr_bigbio_kb schema
- Converts document-level entity annotations to token-level BIO tags
- Builds label mappings and tokenizer preprocessing with aligned labels
- Returns tokenized datasets and data collator for Trainer
"""
from typing import Dict, List, Tuple

import numpy as np
from datasets import ClassLabel, DatasetDict, Features, Sequence, Value, load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
)


def load_bc5cdr_ner(dataset_name: str = "bigbio/bc5cdr", dataset_config: str = "bc5cdr_bigbio_kb") -> DatasetDict:
    """Load BC5CDR in knowledge-base schema.

    The bc5cdr_bigbio_kb config yields documents with passages and entity annotations.
    We'll convert them to token-level BIO format.
    """
    try:
        dset = load_dataset(dataset_name, dataset_config, trust_remote_code=True)
        # Basic sanity check
        for split in ("train", "validation", "test"):
            if split not in dset:
                raise KeyError(f"Split '{split}' missing in dataset {dataset_name}:{dataset_config}")
        required_cols = {"passages", "entities"}
        for split in dset:
            if not required_cols.issubset(set(dset[split].column_names)):
                raise KeyError("Dataset does not expose passages/entities columns.")
        return dset
    except Exception as e:
        raise RuntimeError(
            "Failed to load BC5CDR with bc5cdr_bigbio_kb schema. "
            "Please ensure 'bigbio' datasets are available and use: load_dataset('bigbio/bc5cdr','bc5cdr_bigbio_kb').\n"
            f"Original error: {e}"
        )


def convert_kb_to_tokens_ner(examples: Dict) -> Dict:
    """Convert BigBio KB format (passages, entities) to tokenized NER format.
    
    For each document:
    - Concatenate all passages' text
    - Split into tokens (whitespace)
    - Assign BIO tags based on character offsets of entities
    """
    all_tokens = []
    all_ner_tags = []
    
    for passages, entities in zip(examples["passages"], examples["entities"]):
        # Concatenate passage texts (usually title + abstract)
        full_text = " ".join([p["text"][0] for p in passages])
        
        # Simple whitespace tokenization
        tokens = full_text.split()
        
        # Build a map of token char ranges
        token_spans = []
        char_idx = 0
        for token in tokens:
            start = full_text.find(token, char_idx)
            end = start + len(token)
            token_spans.append((start, end))
            char_idx = end
        
        # Initialize tags as 'O'
        tags = ["O"] * len(tokens)
        
        # Assign BIO tags based on entity offsets
        for ent in entities:
            ent_type = ent["type"]  # e.g., "Chemical", "Disease"
            offsets = ent["offsets"]  # list of [start, end] pairs
            
            for offset_pair in offsets:
                ent_start, ent_end = offset_pair
                
                # Find tokens that overlap with this entity
                for idx, (tok_start, tok_end) in enumerate(token_spans):
                    # Check if token overlaps with entity
                    if tok_start < ent_end and tok_end > ent_start:
                        if tags[idx] == "O":  # not yet tagged
                            if tok_start >= ent_start:  # token starts inside entity
                                if idx > 0 and tags[idx-1].endswith(ent_type):
                                    tags[idx] = f"I-{ent_type}"
                                else:
                                    tags[idx] = f"B-{ent_type}"
                            else:  # token started before entity
                                tags[idx] = f"B-{ent_type}"
        
        all_tokens.append(tokens)
        all_ner_tags.append(tags)
    
    return {"tokens": all_tokens, "ner_tags": all_ner_tags}


def get_label_list(dset: DatasetDict) -> List[str]:
    """Extract unique label strings from the converted dataset."""
    labels = set()
    for ex in dset["train"]["ner_tags"]:
        labels.update(set(ex))
    return sorted(list(labels))


def tokenize_and_align_labels(
    examples: Dict[str, List[List[str]]],
    tokenizer: AutoTokenizer,
    label_all_tokens: bool,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    max_length: int = 256,
) -> Dict[str, List[List[int]]]:
    """Tokenize tokens and align NER labels to subword tokens.

    - For each word, label the first sub-token with the word's tag
    - For remaining sub-tokens, use the same tag if label_all_tokens=True, else -100
    """
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding=False,
        max_length=max_length,
    )

    labels = []
    for i, word_labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
                continue
            # first sub-token of a word
            if word_id != prev_word_id:
                lab = word_labels[word_id]
                # ClassLabel may store ints already
                if isinstance(lab, str):
                    lab_id = label2id[lab]
                else:
                    lab_id = lab
                label_ids.append(lab_id)
            else:
                # subsequent sub-token
                if label_all_tokens:
                    lab = word_labels[word_id]
                    if isinstance(lab, str):
                        lab_id = label2id[lab]
                    else:
                        lab_id = lab
                    label_ids.append(lab_id)
                else:
                    label_ids.append(-100)
            prev_word_id = word_id
        labels.append(label_ids)

    tokenized["labels"] = labels
    return tokenized


def prepare_tokenized_datasets(
    model_name: str,
    dataset_name: str = "bigbio/bc5cdr",
    dataset_config: str = "bc5cdr_bigbio_kb",
    max_length: int = 256,
    label_all_tokens: bool = False,
) -> Tuple[DatasetDict, AutoTokenizer, List[str], Dict[str, int], Dict[int, str], DataCollatorForTokenClassification]:
    """Load dataset, tokenizer, and produce tokenized datasets with aligned labels."""
    raw = load_bc5cdr_ner(dataset_name, dataset_config)

    # Convert from KB format to token + NER tags
    tokenized_raw = raw.map(
        convert_kb_to_tokens_ner,
        batched=True,
        remove_columns=[c for c in raw["train"].column_names if c not in ("id", "document_id")],
        desc="Converting KB to token-level NER",
    )

    # RoBERTa/GPT-2 need add_prefix_space=True for pre-tokenized inputs
    # DeBERTa fast tokenizer has bugs, use slow tokenizer
    tokenizer_kwargs = {}
    if any(name in model_name.lower() for name in ["deberta"]):
        tokenizer_kwargs["use_fast"] = False
        print(f"[DEBUG] Loading {model_name} with slow tokenizer (use_fast=False)")
    else:
        tokenizer_kwargs["use_fast"] = True
        if any(name in model_name.lower() for name in ["roberta", "gpt"]):
            tokenizer_kwargs["add_prefix_space"] = True
    
    print(f"[DEBUG] Tokenizer kwargs for {model_name}: {tokenizer_kwargs}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

    label_list = get_label_list(tokenized_raw)
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}

    def _map_fn(batch):
        return tokenize_and_align_labels(
            batch, tokenizer, label_all_tokens=label_all_tokens,
            label2id=label2id, id2label=id2label, max_length=max_length,
        )

    tokenized = tokenized_raw.map(
        _map_fn,
        batched=True,
        remove_columns=[c for c in tokenized_raw["train"].column_names if c not in ("tokens", "ner_tags")],
        desc="Tokenizing and aligning labels",
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    return tokenized, tokenizer, label_list, label2id, id2label, data_collator
