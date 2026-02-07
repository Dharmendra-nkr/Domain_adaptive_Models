"""
Ensemble Model: Combine predictions from multiple models
- Majority voting across models
- Soft voting (probability averaging)
- Weighted ensemble based on individual F1 scores
"""
import os
import json
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_dataset
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

from src.config import ExperimentConfig


class EnsembleNER:
    def __init__(self, model_configs: List[Dict]):
        """
        Initialize ensemble with multiple models.
        
        Args:
            model_configs: List of dicts with 'name' and 'path' keys
        """
        self.models = []
        self.tokenizers = []
        self.model_names = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Using device: {self.device}")
        print(f"\nLoading {len(model_configs)} models for ensemble...")
        
        for config in model_configs:
            name = config['name']
            path = config['path']
            
            # Find checkpoint
            checkpoint_dir = None
            if os.path.exists(path):
                for item in os.listdir(path):
                    if item.startswith('checkpoint-'):
                        checkpoint_dir = os.path.join(path, item)
                        break
            
            if checkpoint_dir and os.path.exists(checkpoint_dir):
                print(f"  ✓ Loading {name} from {checkpoint_dir}")
                tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
                model = AutoModelForTokenClassification.from_pretrained(checkpoint_dir)
                model.to(self.device)
                model.eval()
                
                self.models.append(model)
                self.tokenizers.append(tokenizer)
                self.model_names.append(name)
            else:
                print(f"  ✗ Skipping {name} (checkpoint not found)")
        
        print(f"\n✓ Loaded {len(self.models)} models successfully\n")
    
    def predict_single_model(self, model, tokenizer, tokens: List[str], max_length=256):
        """Get predictions from a single model."""
        encoding = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        
        # Get predictions and probabilities
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        preds = np.argmax(probs, axis=-1)
        
        # Map back to original tokens
        word_ids = encoding.word_ids()
        token_preds = []
        token_probs = []
        
        for word_id in range(len(tokens)):
            # Find all subword tokens for this word
            word_positions = [i for i, wid in enumerate(word_ids) if wid == word_id]
            if word_positions:
                # Use first subword token's prediction
                token_preds.append(preds[word_positions[0]])
                token_probs.append(probs[word_positions[0]])
        
        return token_preds, token_probs
    
    def predict_majority_voting(self, tokens: List[str]) -> List[int]:
        """Ensemble prediction using majority voting."""
        all_predictions = []
        
        for model, tokenizer in zip(self.models, self.tokenizers):
            preds, _ = self.predict_single_model(model, tokenizer, tokens)
            all_predictions.append(preds)
        
        # Majority vote for each token
        ensemble_preds = []
        num_tokens = len(all_predictions[0])
        
        for token_idx in range(num_tokens):
            votes = [preds[token_idx] for preds in all_predictions if token_idx < len(preds)]
            if votes:
                majority = Counter(votes).most_common(1)[0][0]
                ensemble_preds.append(majority)
        
        return ensemble_preds
    
    def predict_soft_voting(self, tokens: List[str], weights=None) -> List[int]:
        """Ensemble prediction using probability averaging."""
        all_probs = []
        
        for model, tokenizer in zip(self.models, self.tokenizers):
            _, probs = self.predict_single_model(model, tokenizer, tokens)
            all_probs.append(probs)
        
        # Average probabilities
        if weights is None:
            weights = [1.0] * len(self.models)
        
        ensemble_preds = []
        num_tokens = len(all_probs[0])
        
        for token_idx in range(num_tokens):
            # Weighted average of probability distributions
            avg_probs = np.zeros_like(all_probs[0][token_idx])
            total_weight = 0
            
            for probs, weight in zip(all_probs, weights):
                if token_idx < len(probs):
                    avg_probs += probs[token_idx] * weight
                    total_weight += weight
            
            avg_probs /= total_weight
            pred = np.argmax(avg_probs)
            ensemble_preds.append(pred)
        
        return ensemble_preds


def evaluate_ensemble(ensemble: EnsembleNER, dataset, label_list: List[str], method='majority'):
    """Evaluate ensemble on dataset."""
    print(f"Evaluating ensemble using {method} voting...")
    
    all_true_labels = []
    all_pred_labels = []
    
    for i, example in enumerate(dataset):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(dataset)}")
        
        tokens = example['tokens']
        true_labels = example['ner_tags']
        
        # Get ensemble predictions
        if method == 'majority':
            pred_labels = ensemble.predict_majority_voting(tokens)
        else:  # soft voting
            pred_labels = ensemble.predict_soft_voting(tokens)
        
        # Convert to label names
        true_label_names = [label_list[l] for l in true_labels[:len(pred_labels)]]
        pred_label_names = [label_list[p] for p in pred_labels[:len(true_labels)]]
        
        all_true_labels.append(true_label_names)
        all_pred_labels.append(pred_label_names)
    
    # Calculate metrics
    precision = precision_score(all_true_labels, all_pred_labels)
    recall = recall_score(all_true_labels, all_pred_labels)
    f1 = f1_score(all_true_labels, all_pred_labels)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'method': method
    }


def main():
    print("="*80)
    print("ENSEMBLE MODEL - BIOMEDICAL NER")
    print("="*80 + "\n")
    
    cfg = ExperimentConfig()
    
    # Define models for ensemble (use top 4 performers)
    model_configs = [
        {
            'name': 'BioBERT',
            'path': 'models/microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
            'f1': 0.8779
        },
        {
            'name': 'SciBERT',
            'path': 'models/allenai--scibert_scivocab_uncased',
            'f1': 0.8749
        },
        {
            'name': 'RoBERTa',
            'path': 'models/roberta-base',
            'f1': 0.8717
        },
        {
            'name': 'BERT',
            'path': 'models/bert-base-uncased',
            'f1': 0.8433
        }
    ]
    
    # Initialize ensemble
    ensemble = EnsembleNER(model_configs)
    
    if len(ensemble.models) < 2:
        print("❌ Need at least 2 models for ensemble. Exiting.")
        return
    
    # Load test data
    print("Loading BC5CDR test dataset...")
    dataset = load_dataset(cfg.dataset_name, cfg.dataset_config, split='test', trust_remote_code=True)
    
    # Get label list from first model
    label_list = list(ensemble.models[0].config.id2label.values())
    print(f"✓ Loaded {len(dataset)} test examples")
    print(f"✓ Label set: {label_list}\n")
    
    # Evaluate ensemble with different methods
    results = []
    
    print("="*80)
    print("ENSEMBLE EVALUATION")
    print("="*80 + "\n")
    
    # 1. Majority Voting
    print("1. MAJORITY VOTING")
    print("-"*80)
    majority_results = evaluate_ensemble(ensemble, dataset, label_list, method='majority')
    results.append(majority_results)
    print(f"\n  Precision: {majority_results['precision']:.4f}")
    print(f"  Recall:    {majority_results['recall']:.4f}")
    print(f"  F1 Score:  {majority_results['f1']:.4f}\n")
    
    # 2. Soft Voting (equal weights)
    print("2. SOFT VOTING (Equal Weights)")
    print("-"*80)
    soft_results = evaluate_ensemble(ensemble, dataset, label_list, method='soft')
    results.append(soft_results)
    print(f"\n  Precision: {soft_results['precision']:.4f}")
    print(f"  Recall:    {soft_results['recall']:.4f}")
    print(f"  F1 Score:  {soft_results['f1']:.4f}\n")
    
    # 3. Weighted Soft Voting (by individual F1 scores)
    print("3. WEIGHTED SOFT VOTING (by F1)")
    print("-"*80)
    weights = [m['f1'] for m in model_configs if any(n in m['name'] for n in ensemble.model_names)]
    print(f"  Weights: {weights}")
    
    # Modify ensemble for weighted voting
    weighted_preds = []
    weighted_true = []
    
    for i, example in enumerate(dataset):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(dataset)}")
        
        tokens = example['tokens']
        true_labels = example['ner_tags']
        pred_labels = ensemble.predict_soft_voting(tokens, weights=weights)
        
        true_label_names = [label_list[l] for l in true_labels[:len(pred_labels)]]
        pred_label_names = [label_list[p] for p in pred_labels[:len(true_labels)]]
        
        weighted_true.append(true_label_names)
        weighted_preds.append(pred_label_names)
    
    weighted_results = {
        'precision': precision_score(weighted_true, weighted_preds),
        'recall': recall_score(weighted_true, weighted_preds),
        'f1': f1_score(weighted_true, weighted_preds),
        'method': 'weighted_soft'
    }
    results.append(weighted_results)
    
    print(f"\n  Precision: {weighted_results['precision']:.4f}")
    print(f"  Recall:    {weighted_results['recall']:.4f}")
    print(f"  F1 Score:  {weighted_results['f1']:.4f}\n")
    
    # Summary comparison
    print("="*80)
    print("ENSEMBLE RESULTS SUMMARY")
    print("="*80 + "\n")
    
    comparison_df = pd.DataFrame([
        {
            'Model': 'BioBERT (best single)',
            'Precision': 0.8609,
            'Recall': 0.8956,
            'F1': 0.8779
        },
        {
            'Model': 'Ensemble (Majority Vote)',
            'Precision': majority_results['precision'],
            'Recall': majority_results['recall'],
            'F1': majority_results['f1']
        },
        {
            'Model': 'Ensemble (Soft Vote)',
            'Precision': soft_results['precision'],
            'Recall': soft_results['recall'],
            'F1': soft_results['f1']
        },
        {
            'Model': 'Ensemble (Weighted)',
            'Precision': weighted_results['precision'],
            'Recall': weighted_results['recall'],
            'F1': weighted_results['f1']
        }
    ])
    
    print(comparison_df.to_string(index=False))
    
    # Find best method
    best_method = max(results, key=lambda x: x['f1'])
    best_single = 0.8779
    improvement = best_method['f1'] - best_single
    
    print(f"\n{'='*80}")
    if improvement > 0:
        print(f"✓ ENSEMBLE IMPROVEMENT: +{improvement*100:.2f}% over best single model")
        print(f"  Best ensemble method: {best_method['method']}")
        print(f"  Ensemble F1: {best_method['f1']:.4f}")
    else:
        print(f"⚠ Ensemble did not improve over best single model")
        print(f"  Best single F1: {best_single:.4f}")
        print(f"  Best ensemble F1: {best_method['f1']:.4f}")
        print(f"  Difference: {improvement*100:.2f}%")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_path = 'results/ensemble_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to: {results_path}")
    
    comparison_df.to_csv('results/ensemble_comparison.csv', index=False)
    print(f"✓ Comparison saved to: results/ensemble_comparison.csv")
    
    print(f"\n{'='*80}")
    print("✅ ENSEMBLE MODEL EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
