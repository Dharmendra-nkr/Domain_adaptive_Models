"""Pre-download models to avoid network issues during training."""
from transformers import AutoTokenizer, AutoModel

models = ["bert-base-uncased", "roberta-base", "microsoft/deberta-base"]

for model_name in models:
    print(f"Downloading {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        print(f"✓ {model_name} downloaded")
    except Exception as e:
        print(f"✗ {model_name} failed: {e}")
