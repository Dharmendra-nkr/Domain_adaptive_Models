"""
Error Analysis: Identify and analyze model failures on BC5CDR test set.
- Extracts misclassifications
- Analyzes error patterns
- Identifies hard cases
- Generates visualizations
"""
import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

from src.data_utils import prepare_tokenized_datasets
from src.config import ExperimentConfig


def load_model_and_tokenizer(model_name: str, model_path: str):
    """Load model and tokenizer from saved checkpoint."""
    print(f"Loading model: {model_path}")
    # Find the checkpoint directory
    checkpoint_dir = None
    for item in os.listdir(model_path):
        if item.startswith('checkpoint-'):
            checkpoint_dir = os.path.join(model_path, item)
            print(f"  Found checkpoint: {checkpoint_dir}")
            break
    
    if checkpoint_dir is None:
        print(f"  No checkpoint found, using model directory directly")
        checkpoint_dir = model_path
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForTokenClassification.from_pretrained(checkpoint_dir)
    return model, tokenizer


def get_predictions(model, tokenizer, dataset, label_list: List[str], device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Generate predictions for entire dataset."""
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_texts = []
    
    print(f"Generating predictions on {len(dataset)} samples...")
    
    with torch.no_grad():
        for i, example in enumerate(dataset):
            if i % 100 == 0:
                print(f"  {i}/{len(dataset)}")
            
            # Get tokens and labels
            tokens = example['tokens']
            labels = example['ner_tags']
            
            # Tokenize
            encoding = tokenizer(
                tokens,
                truncation=True,
                is_split_into_words=True,
                padding='max_length',
                max_length=256,
                return_tensors='pt'
            )
            
            # Forward pass
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Get predictions
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()[0]
            
            # Filter valid tokens (not padding, not [CLS], not [SEP])
            word_ids = encoding.word_ids()
            valid_preds = []
            valid_labels = []
            valid_tokens = []
            
            for token_idx, word_id in enumerate(word_ids):
                if word_id is not None and word_id < len(tokens):
                    valid_preds.append(preds[token_idx])
                    if word_id < len(labels):
                        valid_labels.append(labels[word_id])
                        valid_tokens.append(tokens[word_id])
            
            all_preds.append(valid_preds)
            all_labels.append(valid_labels)
            all_texts.append(tokens)
    
    return all_preds, all_labels, all_texts


def extract_entities(tokens: List[str], labels: List[int], label_list: List[str]) -> List[Dict]:
    """Extract entities from token-level predictions."""
    entities = []
    current_entity = None
    
    for token, label_id in zip(tokens, labels):
        label = label_list[label_id]
        
        if label == 'O':  # Outside entity
            if current_entity:
                entities.append(current_entity)
                current_entity = None
        elif label.startswith('B-'):  # Begin entity
            if current_entity:
                entities.append(current_entity)
            entity_type = label[2:]
            current_entity = {
                'type': entity_type,
                'tokens': [token],
                'text': token
            }
        elif label.startswith('I-'):  # Inside entity
            if current_entity:
                current_entity['tokens'].append(token)
                current_entity['text'] += ' ' + token
    
    if current_entity:
        entities.append(current_entity)
    
    return entities


def analyze_errors(all_preds, all_labels, all_texts, label_list: List[str]):
    """Analyze prediction errors."""
    errors = []
    error_types = defaultdict(int)
    correct_count = 0
    total_count = 0
    
    print("Analyzing errors...")
    
    for pred_seq, label_seq, text_seq in zip(all_preds, all_labels, all_texts):
        # Ensure same length
        min_len = min(len(pred_seq), len(label_seq))
        pred_seq = pred_seq[:min_len]
        label_seq = label_seq[:min_len]
        text_seq = text_seq[:min_len]
        
        pred_entities = extract_entities(text_seq, pred_seq, label_list)
        true_entities = extract_entities(text_seq, label_seq, label_list)
        
        # Token-level errors
        for token, pred_label_id, true_label_id in zip(text_seq, pred_seq, label_seq):
            true_label = label_list[true_label_id]
            pred_label = label_list[pred_label_id]
            total_count += 1
            
            if pred_label_id == true_label_id:
                correct_count += 1
            else:
                error_types[f"{true_label} → {pred_label}"] += 1
                errors.append({
                    'token': token,
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'context': ' '.join(text_seq),
                    'sentence': text_seq
                })
    
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    return {
        'errors': errors,
        'error_types': error_types,
        'accuracy': accuracy,
        'total_tokens': total_count,
        'correct_tokens': correct_count
    }


def generate_error_report(analysis_results: Dict, model_name: str, output_dir: str = 'results'):
    """Generate error analysis report."""
    os.makedirs(output_dir, exist_ok=True)
    
    errors = analysis_results['errors']
    error_types = analysis_results['error_types']
    accuracy = analysis_results['accuracy']
    total = analysis_results['total_tokens']
    
    # Sort error types by frequency
    sorted_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)
    
    report = f"""
================================================================================
ERROR ANALYSIS REPORT: {model_name}
================================================================================

OVERALL STATISTICS
==================
Total Tokens: {total}
Correct Tokens: {analysis_results['correct_tokens']}
Token-Level Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)
Total Error Types: {len(error_types)}

TOP 10 ERROR PATTERNS
=====================
"""
    
    for i, (error_type, count) in enumerate(sorted_errors[:10], 1):
        percentage = (count / total) * 100
        report += f"{i:2d}. {error_type:30s} - {count:4d} errors ({percentage:.2f}%)\n"
    
    # Find most common misclassified tokens
    token_errors = Counter([e['token'] for e in errors])
    
    report += f"\n\nMOST COMMONLY MISCLASSIFIED TOKENS (Top 20)\n"
    report += "=" * 60 + "\n"
    for token, count in token_errors.most_common(20):
        report += f"  '{token}' - {count} misclassifications\n"
    
    # Sample errors
    report += f"\n\nSAMPLE ERRORS (First 30)\n"
    report += "=" * 60 + "\n"
    for i, error in enumerate(errors[:30], 1):
        report += f"\n{i}. Token: '{error['token']}'\n"
        report += f"   True Label: {error['true_label']}\n"
        report += f"   Pred Label: {error['pred_label']}\n"
        report += f"   Context: ...{error['context'][:100]}...\n"
    
    report += f"\n{'='*80}\nReport generated on BC5CDR test set\n"
    
    return report


def main():
    cfg = ExperimentConfig()
    
    # Use best model (BioBERT)
    best_model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    best_model_path = os.path.join(cfg.output_dir, best_model_name.replace("/", "--"))
    
    print("="*80)
    print("ERROR ANALYSIS - BIOMEDICAL NER")
    print("="*80)
    print(f"\nAnalyzing: {best_model_name}")
    print(f"Model Path: {best_model_path}\n")
    
    # Load data
    tokenized, tokenizer, label_list, label2id, id2label, data_collator = prepare_tokenized_datasets(
        model_name=best_model_name,
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        max_length=cfg.max_length,
        label_all_tokens=cfg.label_all_tokens,
    )
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(best_model_name, best_model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Get predictions on test set
    all_preds, all_labels, all_texts = get_predictions(
        model, tokenizer, tokenized['test'], label_list, device=device
    )
    
    # Analyze errors
    analysis = analyze_errors(all_preds, all_labels, all_texts, label_list)
    
    # Generate report
    report = generate_error_report(analysis, best_model_name)
    print(report)
    
    # Save report
    report_path = os.path.join('results', f'error_analysis_{best_model_name.replace("/", "_")}.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n✓ Report saved to: {report_path}")
    
    # Save detailed errors as JSON
    errors_json_path = os.path.join('results', f'errors_{best_model_name.replace("/", "_")}.json')
    errors_for_json = [
        {
            'token': e['token'],
            'true_label': e['true_label'],
            'pred_label': e['pred_label'],
            'context': e['context']
        }
        for e in analysis['errors']
    ]
    
    with open(errors_json_path, 'w') as f:
        json.dump({
            'model': best_model_name,
            'total_errors': len(analysis['errors']),
            'accuracy': analysis['accuracy'],
            'error_types': dict(analysis['error_types']),
            'sample_errors': errors_for_json[:100]
        }, f, indent=2)
    print(f"✓ Detailed errors saved to: {errors_json_path}")
    
    # Save error statistics
    stats_df = pd.DataFrame([
        {'error_type': k, 'count': v}
        for k, v in sorted(analysis['error_types'].items(), key=lambda x: x[1], reverse=True)
    ])
    
    stats_path = os.path.join('results', f'error_stats_{best_model_name.replace("/", "_")}.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"✓ Error statistics saved to: {stats_path}")
    
    print(f"\n✅ Error analysis complete!")
    print(f"\nKey Findings:")
    print(f"  • Token-level accuracy: {analysis['accuracy']*100:.2f}%")
    print(f"  • Total errors: {len(analysis['errors'])}")
    print(f"  • Most common error: {sorted(analysis['error_types'].items(), key=lambda x: x[1], reverse=True)[0]}")


if __name__ == "__main__":
    main()
