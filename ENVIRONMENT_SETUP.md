# Environment Setup Guide

## ⚠️ Important: Activate Conda Environment First!

You have PyTorch installed in a conda environment, but you're currently in the `base` environment.

### Available Environments:
- `pytorch-gpu` - Has PyTorch with GPU support
- `biomedical-ner` - Your NER project environment  
- `base` - Default conda (no PyTorch)

---

## Quick Fix: Activate Environment

```powershell
# Option 1: Use pytorch-gpu environment
conda activate pytorch-gpu

# Option 2: Use biomedical-ner environment
conda activate biomedical-ner

# Then run DAPT
python -m src.dapt_pretraining
```

---

## Verify GPU Setup

After activating the environment, verify everything works:

```powershell
# Check PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check GPU name
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"
```

Expected output:
```
PyTorch: 2.x.x
CUDA available: True
GPU: NVIDIA GeForce RTX ... (or your GPU model)
```

---

## If PyTorch Not Installed in Any Environment

Install PyTorch with CUDA support:

```powershell
# Create new environment (recommended)
conda create -n biomedical-ner-gpu python=3.11 -y
conda activate biomedical-ner-gpu

# Install PyTorch with CUDA (check pytorch.org for latest)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install other dependencies
pip install transformers datasets seqeval accelerate pandas scikit-learn matplotlib seaborn
```

---

## Run DAPT (After Activation)

```powershell
# Make sure you're in the right environment
conda activate pytorch-gpu  # or biomedical-ner

# Navigate to project
cd "d:\OpenLab 3\biomedical-ner-dapt"

# Run DAPT
python -m src.dapt_pretraining
```

---

## Troubleshooting

### "No module named 'torch'"
→ You're in the wrong conda environment. Activate `pytorch-gpu` or `biomedical-ner`

### "CUDA not available"
→ PyTorch installed without CUDA. Reinstall with conda command above

### "Out of memory"
→ Reduce batch size: `python -m src.dapt_pretraining --batch-size 8`

---

## Next Steps

1. **Activate environment**: `conda activate pytorch-gpu`
2. **Verify GPU**: Run verification commands above
3. **Run DAPT**: `python -m src.dapt_pretraining`
