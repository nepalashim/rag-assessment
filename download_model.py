"""
Pre-download the sentence-transformers embedding model.

Note: This is for EMBEDDINGS only, NOT for LLM.
Since I am using Groq, the LLM runs in the cloud (no download needed).
But embeddings run locally, so we need to download this model.
"""

print("Starting download script...")
print("=" * 60)

try:
    print("Importing sentence_transformers...")
    from sentence_transformers import SentenceTransformer
    print(" Import successful!")
except ImportError as e:
    print(f" Failed to import sentence_transformers: {e}")
    print("\nPlease install it:")
    print("  pip install sentence-transformers")
    exit(1)

import os

print("=" * 60)
print("DOWNLOADING EMBEDDING MODEL")
print("=" * 60)
print("\n  Important:")
print("   - This downloads the EMBEDDING model (runs locally)")
print("   - Your LLM (Groq) runs in the cloud (no download needed)")
print("   - Just need this embedding model (~80 MB)")
print()

model_name = "sentence-transformers/all-MiniLM-L6-v2"
print(f"Model: {model_name}")
print("Size: ~80 MB")
print("Purpose: Convert text to vector embeddings")
print("\nThis will take 1-2 minutes...")
print("=" * 60)
print()

try:
    # Download and load the model
    print("Downloading... (this happens only once)")
    model = SentenceTransformer(model_name)
    
    print("\n Model downloaded successfully!")
    
    # Test the model
    print("\n Testing model with sample text...")
    test_text = "This is a test sentence for embedding generation."
    embedding = model.encode(test_text)
    
    print(f"Generated embedding of dimension: {len(embedding)}")
    print(f" Embedding model is ready to use!")
    
    # Show cache location
    cache_dir = os.path.expanduser("~/.cache/torch/sentence_transformers/")
    print(f"\n Model cached at:")
    print(f"   {cache_dir}")
    
    print("\n" + "=" * 60)
    print(" SETUP COMPLETE!")
    print("=" * 60)
    print("\nYour setup:")
    print("  • Embeddings: sentence-transformers (local) ")
    print("  • LLM: Groq (cloud) ")
    print("  • No more downloads needed!")
    
except Exception as e:
    print(f"\n Error downloading model: {e}")
    print("\nTroubleshooting:")
    print("  1. Check internet connection")
    print("  2. Make sure you have ~100 MB free disk space")
    print("  3. Try again: python download_model.py")

print("\n" + "=" * 60)