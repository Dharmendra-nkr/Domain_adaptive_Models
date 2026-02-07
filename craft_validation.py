#!/usr/bin/env python3
"""
CRAFT Dataset - Cross-Dataset Validation
Evaluates optimized BioBERT on CRAFT (Concept Recognition and Annotations on Full-Text) corpus
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

# Configuration
MODEL_PATH = "models/biobert_optimized/checkpoint-945"
DATASET_NAME = "craft"
RESULTS_DIR = Path("results/craft_validation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_model_and_tokenizer():
    """Load optimized BioBERT model"""
    print(f"\n📦 Loading model from {MODEL_PATH}...")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_safetensors=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully")
    return model, tokenizer

def load_craft_dataset():
    """Load CRAFT dataset from BigBio"""
    print("\n📥 Loading CRAFT dataset...")
    try:
        # CRAFT dataset from BigBio
        dataset = load_dataset("craft")
        print(f"✅ Dataset loaded: {dataset}")
        return dataset
    except Exception as e:
        print(f"⚠️ Error loading CRAFT dataset: {e}")
        print("Note: CRAFT is a full-text biomedical corpus with higher complexity")
        return None

def evaluate_dataset(model, tokenizer, dataset, split="test"):
    """Evaluate model on CRAFT dataset"""
    print(f"\n🔍 Evaluating on {split} set...")
    
    if not dataset or split not in dataset:
        print(f"⚠️ Dataset split '{split}' not found")
        # Try to use available split
        if "validation" in dataset:
            split = "validation"
        elif "test" in dataset:
            split = "test"
        else:
            split = list(dataset.keys())[0]
        print(f"Using '{split}' split instead")
    
    data = dataset[split]
    print(f"Processing {len(data)} documents...")
    
    predictions = []
    true_labels = []
    
    start_time = time.time()
    
    for idx, example in enumerate(data):
        if idx % 50 == 0:
            print(f"  Processed {idx}/{len(data)} documents...")
        
        # Get tokens and labels
        tokens = example.get("tokens", [])
        ner_tags = example.get("ner_tags", [])
        
        if not tokens or not ner_tags:
            continue
        
        # Tokenize
        encoding = tokenizer(
            tokens,
            truncation=True,
            max_length=512,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding=False
        )
        
        # Prepare aligned labels
        aligned_labels = align_labels_with_tokens(ner_tags, encoding, tokenizer)
        
        # Model prediction
        input_ids = torch.tensor([encoding["input_ids"]]).to(device)
        attention_mask = torch.tensor([encoding["attention_mask"]]).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        
        # Get predictions
        pred_ids = torch.argmax(logits, dim=2)[0].cpu().numpy()
        
        # Convert to label names
        pred_labels = [model.config.id2label[int(id_)] for id_ in pred_ids if int(id_) != -100]
        true_labels_sample = [model.config.id2label[int(label)] for label in aligned_labels if int(label) != -100]
        
        if pred_labels and true_labels_sample:
            predictions.append(pred_labels[:len(true_labels_sample)])
            true_labels.append(true_labels_sample)
    
    elapsed = time.time() - start_time
    print(f"✅ Evaluation completed in {elapsed:.1f} seconds")
    
    return {
        "predictions": predictions,
        "true_labels": true_labels,
        "elapsed_time": elapsed,
        "split": split
    }

def align_labels_with_tokens(labels, encoding, tokenizer):
    """Align labels with tokenized input"""
    aligned_labels = []
    
    for i, word_id in enumerate(encoding.word_ids()):
        if word_id is None:
            aligned_labels.append(-100)
        else:
            if word_id < len(labels):
                aligned_labels.append(labels[word_id])
            else:
                aligned_labels.append(-100)
    
    return aligned_labels

def calculate_metrics(predictions, true_labels) -> Dict:
    """Calculate NER metrics"""
    print("\n📊 Calculating metrics...")
    
    metrics = {
        "f1": f1_score(true_labels, predictions),
        "precision": precision_score(true_labels, predictions),
        "recall": recall_score(true_labels, predictions),
        "report": classification_report(true_labels, predictions)
    }
    
    return metrics

def save_results(eval_data, metrics):
    """Save evaluation results"""
    print("\n💾 Saving results...")
    
    # JSON results
    results_json = {
        "dataset": "CRAFT",
        "model": "BioBERT (Optimized Checkpoint 945)",
        "split": eval_data["split"],
        "metrics": {
            "f1": float(metrics["f1"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"])
        },
        "samples_evaluated": len(eval_data["true_labels"]),
        "evaluation_time_seconds": eval_data["elapsed_time"],
        "status": "SUCCESS"
    }
    
    results_file = RESULTS_DIR / "craft_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)
    print(f"✅ Results saved to {results_file}")
    
    # CSV format
    csv_file = RESULTS_DIR / "craft_metrics.csv"
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("Dataset,F1,Precision,Recall,Test_Size\n")
        f.write(f"CRAFT,{metrics['f1']:.4f},{metrics['precision']:.4f},{metrics['recall']:.4f},{len(eval_data['true_labels'])}\n")
    print(f"✅ CSV saved to {csv_file}")
    
    # Detailed report
    report_file = RESULTS_DIR / "craft_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# CRAFT Dataset - Evaluation Report\n\n")
        f.write(f"Dataset: CRAFT (Full-text Biomedical Corpus)\n")
        f.write(f"Model: BioBERT (Optimized Checkpoint 945)\n")
        f.write(f"Split: {eval_data['split']}\n")
        f.write(f"Evaluation Time: {eval_data['elapsed_time']:.1f}s\n")
        f.write(f"Documents Evaluated: {len(eval_data['true_labels'])}\n\n")
        f.write("## Metrics\n\n")
        f.write(f"F1 Score:   {metrics['f1']:.4f}\n")
        f.write(f"Precision:  {metrics['precision']:.4f}\n")
        f.write(f"Recall:     {metrics['recall']:.4f}\n\n")
        f.write("## Analysis\n\n")
        f.write("Note: CRAFT is a full-text corpus (vs BC5CDR abstracts)\n")
        f.write("- Multiple entity types across full papers\n")
        f.write("- Higher complexity due to document length\n")
        f.write("- Domain includes: Gene/Protein, Chemicals, Organisms, Anatomical entities\n\n")
        f.write("## Detailed Classification Report\n\n")
        f.write(str(metrics['report']))
    print(f"✅ Report saved to {report_file}")
    
    return results_json

def main():
    print("\n" + "="*60)
    print("CRAFT DATASET - CROSS-DATASET VALIDATION")
    print("(Full-text Biomedical Corpus)")
    print("="*60)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer()
    
    # Load dataset
    try:
        dataset = load_craft_dataset()
        
        if dataset is None:
            print("\n⚠️ CRAFT dataset could not be loaded")
            print("This dataset requires internet connection and may need additional setup")
            return
        
        # Evaluate
        eval_data = evaluate_dataset(model, tokenizer, dataset)
        
        if eval_data is None or not eval_data["predictions"]:
            print("⚠️ Evaluation failed")
            return
        
        # Calculate metrics
        metrics = calculate_metrics(eval_data["predictions"], eval_data["true_labels"])
        
        # Display results
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"\nDataset: CRAFT (Full-text Biomedical Corpus)")
        print(f"Documents Evaluated: {len(eval_data['true_labels'])}")
        print(f"\nF1 Score:   {metrics['f1']:.4f}")
        print(f"Precision:  {metrics['precision']:.4f}")
        print(f"Recall:     {metrics['recall']:.4f}")
        print(f"\nExpected F1: 78-85% (higher complexity than abstracts)")
        print(f"Actual F1:   {metrics['f1']*100:.2f}%")
        print("="*60 + "\n")
        
        # Save results
        results = save_results(eval_data, metrics)
        
        print("\n✅ CRAFT validation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
