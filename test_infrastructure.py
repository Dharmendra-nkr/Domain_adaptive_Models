"""
Test script to verify DAPT and cross-dataset infrastructure.
Runs quick validation without full training.
"""
import sys
import os

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from src.dapt_config import DAPTConfig, CROSS_DATASETS
        from src.dapt_utils import load_pubmed_corpus, clean_biomedical_text
        from src.dapt_pretraining import run_dapt
        from src.run_cross_dataset import run_cross_dataset_experiments
        from src.plot_cross_dataset import generate_all_cross_dataset_plots
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    try:
        from src.dapt_config import DAPTConfig, CROSS_DATASETS
        
        cfg = DAPTConfig()
        print(f"  DAPT config: {cfg.base_model}, {cfg.num_train_epochs} epochs")
        print(f"  Cross-datasets: {len(CROSS_DATASETS)} datasets")
        for ds in CROSS_DATASETS:
            print(f"    - {ds.display_name}: {ds.name}")
        print("✓ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_text_cleaning():
    """Test text cleaning utilities."""
    print("\nTesting text cleaning...")
    try:
        from src.dapt_utils import clean_biomedical_text
        
        test_cases = [
            "Aspirin   is  a  common   drug.",
            "Cancer (malignant neoplasm) affects millions.",
            "  Leading/trailing   spaces  ",
        ]
        
        for text in test_cases:
            cleaned = clean_biomedical_text(text)
            print(f"  '{text}' -> '{cleaned}'")
        
        print("✓ Text cleaning works")
        return True
    except Exception as e:
        print(f"✗ Text cleaning test failed: {e}")
        return False


def test_dataset_loading():
    """Test dataset loading (small sample)."""
    print("\nTesting dataset loading...")
    try:
        from src.dapt_utils import load_pubmed_corpus
        
        print("  Attempting to load small PubMed sample...")
        # Try to load a very small sample
        dataset = load_pubmed_corpus(
            corpus_name="scientific_papers",
            subset="train[:100]",
            validation_split=0.1
        )
        
        print(f"  Train examples: {len(dataset['train'])}")
        print(f"  Validation examples: {len(dataset['validation'])}")
        print("✓ Dataset loading works")
        return True
    except Exception as e:
        print(f"✗ Dataset loading test failed: {e}")
        print("  This is expected if you don't have internet connection")
        print("  or the dataset is not available. It will work during actual training.")
        return False


def test_cross_dataset_config():
    """Test cross-dataset configuration."""
    print("\nTesting cross-dataset configuration...")
    try:
        from src.dapt_config import CROSS_DATASETS
        
        print(f"  Available datasets: {len(CROSS_DATASETS)}")
        for ds in CROSS_DATASETS:
            print(f"    {ds.display_name}:")
            print(f"      Name: {ds.name}")
            print(f"      Config: {ds.config}")
            print(f"      Description: {ds.description}")
        
        print("✓ Cross-dataset configuration valid")
        return True
    except Exception as e:
        print(f"✗ Cross-dataset test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("DAPT & Cross-Dataset Infrastructure Test")
    print("="*70)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Text Cleaning", test_text_cleaning),
        ("Cross-Dataset Config", test_cross_dataset_config),
        ("Dataset Loading", test_dataset_loading),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            results.append((name, False))
    
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Infrastructure is ready.")
        print("\nNext steps:")
        print("  1. Run DAPT: python -m src.dapt_pretraining")
        print("  2. Run cross-dataset: python -m src.run_cross_dataset --models bert-base-uncased microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        if passed >= 3:
            print("   Most core functionality works - failures may be due to missing datasets.")
    
    print("="*70)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
