import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
import argparse
import time

def get_model():
    model_name = "TaylorAI/bge-micro-v2"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    # Note: We assume the model is already downloaded by search_vault.py or the daemon
    return SentenceTransformer(model_name, local_files_only=True)

def index_directory(directory_path, index_name, central_store):
    model = get_model()
    paths = []
    texts = []
    
    print(f"Scanning {directory_path}...")
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(('.md', '.txt', '.py', '.js', '.ts', '.html', '.css', '.json')):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if content.strip():
                            # For now, we index the whole file as one chunk
                            # In the future, we could split by headings or paragraphs
                            paths.append(file_path)
                            texts.append(content)
                except:
                    continue
    
    if not paths:
        print("No indexable files found.")
        return

    print(f"Generating embeddings for {len(paths)} files...")
    start_time = time.time()
    embeddings = model.encode(texts, show_progress_bar=True)
    duration = time.time() - start_time
    
    output_path = os.path.join(central_store, f"{index_name}.npz")
    os.makedirs(central_store, exist_ok=True)
    
    np.savez_compressed(output_path, paths=np.array(paths), vectors=embeddings)
    print(f"Index '{index_name}' saved to {output_path} in {duration:.2f}s")

if __name__ == "__main__":
    default_store = os.environ.get("SMART_SEARCH_INDICES", os.path.expanduser("~/.smart-search/indices"))
    
    parser = argparse.ArgumentParser(description="Standalone Indexer for Global Semantic Search")
    parser.add_argument("directory_path", help="Directory to index")
    parser.add_argument("index_name", help="Name for the resulting index file")
    parser.add_argument("--store", default=default_store, help="Central index storage path")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory_path):
        print(f"Error: {args.directory_path} is not a directory.")
        sys.exit(1)
        
    index_directory(args.directory_path, args.index_name, args.store)
