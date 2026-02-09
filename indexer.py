import os
import sys
import time
import argparse

import numpy as np

from shared import (
    CENTRAL_INDEX_STORE,
    INDEXABLE_EXTENSIONS,
    MAX_FILE_SIZE,
    SKIP_DIRS,
    chunk_text,
    get_model,
    try_daemon_reload,
)


def index_directory(directory_path, index_name, central_store):
    model = get_model()
    file_paths = []
    file_texts = []

    print(f"Scanning {directory_path}...")
    for root, dirs, files in os.walk(directory_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for file in files:
            if file.endswith(INDEXABLE_EXTENSIONS):
                file_path = os.path.join(root, file)
                try:
                    if os.path.getsize(file_path) > MAX_FILE_SIZE:
                        print(f"Skipping large file ({os.path.getsize(file_path)} bytes): {file_path}")
                        continue
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                        if content.strip():
                            file_paths.append(file_path)
                            file_texts.append(content)
                except OSError:
                    continue

    if not file_paths:
        print("No indexable files found.")
        return

    # Chunk files into smaller segments for sharper embeddings
    chunk_paths = []
    chunk_texts = []
    for fpath, ftext in zip(file_paths, file_texts):
        chunks = chunk_text(ftext)
        for chunk in chunks:
            chunk_paths.append(fpath)
            chunk_texts.append(chunk)

    print(f"Generating embeddings for {len(file_paths)} files ({len(chunk_texts)} chunks)...")
    start_time = time.time()
    embeddings = model.encode(chunk_texts, show_progress_bar=True)
    duration = time.time() - start_time

    output_path = os.path.join(central_store, f"{index_name}.npz")
    os.makedirs(central_store, exist_ok=True)

    np.savez_compressed(output_path, paths=np.array(chunk_paths), vectors=embeddings)
    print(f"Index '{index_name}' saved to {output_path} in {duration:.2f}s")
    print(f"  {len(file_paths)} files → {len(chunk_texts)} chunks")

    if try_daemon_reload():
        print("Notified Search Booster to reload indices.")


if __name__ == "__main__":
    default_store = CENTRAL_INDEX_STORE

    parser = argparse.ArgumentParser(description="Standalone Indexer for Global Semantic Search")
    parser.add_argument("directory_path", nargs="?", help="Directory to index (optional if using --remove)")
    parser.add_argument("index_name", nargs="?", help="Name for the index file")
    parser.add_argument("--store", default=default_store, help="Central index storage path")
    parser.add_argument("--remove", help="Remove an index by name")

    args = parser.parse_args()

    if args.remove:
        output_path = os.path.join(args.store, f"{args.remove}.npz")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                print(f"Index '{args.remove}' removed from {output_path}")
                if try_daemon_reload():
                    print("Notified Search Booster to reload indices.")
            except OSError as e:
                print(f"Error removing index: {e}")
        else:
            print(f"Index '{args.remove}' not found at {output_path}")
        sys.exit(0)

    if not args.directory_path or not args.index_name:
        parser.print_help()
        sys.exit(1)

    if not os.path.isdir(args.directory_path):
        print(f"Error: {args.directory_path} is not a directory.")
        sys.exit(1)

    index_directory(args.directory_path, args.index_name, args.store)
