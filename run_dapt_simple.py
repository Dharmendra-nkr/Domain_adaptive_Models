#!/usr/bin/env python
"""Run DAPT with small corpus for testing."""
from src.dapt_config import DAPTConfig
from src.dapt_pretraining import run_dapt
import os

print("Starting DAPT with small corpus...")
cfg = DAPTConfig(
    corpus_subset="train[:10000]",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    logging_steps=200,
    output_dir="models/bert-base-dapt"
)

try:
    run_dapt(cfg)
    print("\n✓ DAPT completed successfully")
    if os.path.exists(cfg.output_dir):
        files = os.listdir(cfg.output_dir)
        print(f"✓ Model saved to {cfg.output_dir}")
        print(f"  Files: {files[:5]}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
