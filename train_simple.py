#!/usr/bin/env python
"""
Simple runner that avoids verbose transformers initialization.
"""
import sys
import os

# Suppress transformers logging during init
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['HF_DATASETS_DISABLE_PROGRESS_BAR'] = '1'

if __name__ == "__main__":
    from src.config import ExperimentConfig
    from src.train import train_single_model
    
    cfg = ExperimentConfig()
    print(f"Starting training: {cfg.model_names[0]}")
    res = train_single_model(cfg.model_names[0], cfg)
    print("Training complete!")
    print(res)
