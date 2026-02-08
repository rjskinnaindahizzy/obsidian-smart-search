import os
import json
import sys
import numpy as np
import argparse
import socket

# Constants
DAEMON_PORT = 5555
DAEMON_HOST = '127.0.0.1'
# Use a generic path for the public repo, or allow override via env var
CENTRAL_INDEX_STORE = os.environ.get("SMART_SEARCH_INDICES", os.path.expanduser("~/.smart-search/indices"))

def get_model():
    from sentence_transformers import SentenceTransformer
    model_name = "TaylorAI/bge-micro-v2"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    return SentenceTransformer(model_name, local_files_only=True)

def try_daemon_search(query, scope=None, index=None, threshold=0.6):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            s.connect((DAEMON_HOST, DAEMON_PORT))
            request = {
                "command": "search", 
                "query": query,
                "scope": scope,
                "index": index,
                "threshold": threshold
            }
            s.sendall(json.dumps(request).encode('utf-8'))
            response = s.recv(131072).decode('utf-8')
            return json.loads(response)
    except Exception as e:
        # print(f"DEBUG: Daemon connection failed: {e}", file=sys.stderr)
        return None

def try_daemon_reload():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            s.connect((DAEMON_HOST, DAEMON_PORT))
            s.sendall(json.dumps({"command": "reload"}).encode('utf-8'))
            return True
    except:
        return False

def cosine_similarity(query_vec, target_vecs):
    dot_product = np.dot(target_vecs, query_vec)
    norms = np.linalg.norm(target_vecs, axis=1) * np.linalg.norm(query_vec)
    return dot_product / norms

def refresh_cache(vault_path, cache_path):
    multi_path = os.path.join(vault_path, ".smart-env", "multi")
    
    paths = []
    vectors = []
    
    print(f"Aggregating embeddings from {multi_path}...")
    if not os.path.exists(multi_path):
        print(f"Error: Multi-path not found at {multi_path}")
        return

    for filename in os.listdir(multi_path):
        if filename.endswith(".ajson"):
            file_path = os.path.join(multi_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    if line.endswith(","): line = line[:-1]
                    try:
                        content = "{" + line + "}"
                        data = json.loads(content)
                        for key, val in data.items():
                            if isinstance(val, dict) and "embeddings" in val:
                                embeds = val["embeddings"]
                                found_vec = None
                                for mod_key, mod_val in embeds.items():
                                    if "TaylorAI" in mod_key and "vec" in mod_val:
                                        found_vec = mod_val["vec"]
                                        break
                                
                                if found_vec:
                                    path = val.get("path") or key.replace("smart_sources:", "").replace("smart_blocks:", "")
                                    paths.append(path)
                                    vectors.append(found_vec)
                    except: continue
    
    np.savez_compressed(cache_path, paths=np.array(paths), vectors=np.array(vectors))
    print(f"Cache saved to {cache_path} ({len(paths)} vectors)")
    
    if try_daemon_reload():
        print("Notified Search Booster to reload.")

def search_indexed_files(query, indices, top_k=20, threshold=0.6, scope=None):
    all_results = []
    model = None # Lazy load
    query_vec = None
    
    for label, cache_path in indices:
        if not os.path.exists(cache_path):
            continue
        
        try:
            data = np.load(cache_path)
            paths = data["paths"]
            vectors = data["vectors"]
            
            if model is None:
                model = get_model()
                query_vec = model.encode(query)
                
            scores = cosine_similarity(query_vec, vectors)
            
            for i, score in enumerate(scores):
                path = str(paths[i])
                if score >= threshold:
                    if not scope or scope.lower() in path.lower():
                        all_results.append({"path": path, "score": float(score), "index": label})
        except Exception as e:
            print(f"Error loading {label}: {e}", file=sys.stderr)
                    
    all_results.sort(key=lambda x: x["score"], reverse=True)
    
    # Deduplicate
    seen = set()
    unique = []
    for r in all_results:
        if r["path"] not in seen:
            unique.append(r)
            seen.add(r["path"])
        if len(unique) >= top_k: break
        
    return unique

def print_help():
    help_text = """
Obsidian Semantic Search (Global)
================================

Usage:
  obs-search "<query>" [options]

Options:
  --scope <folder>   Filter results to paths containing this string.
  --index <name>     Search only a specific index (e.g., 'vault', 'documents').
  --refresh          Rebuild the Obsidian vault cache.
  --threshold <val>  Similarity threshold (0.0-1.0, default 0.6).
  -h, --help         Show this help menu.

Booster Commands:
  obs-search-server  Start background booster (sub-second search).
  obs-search-stop    Stop background booster.

External Indexing:
  obs-index <path> <name>   Index a new external directory.
    """
    print(help_text.strip())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("query", nargs="*", help="Search query")
    parser.add_argument("--vault_path", default="e:/Obsidian Vault", help="Path to Obsidian vault")
    parser.add_argument("--scope", help="Optional folder/path substring to scope search")
    parser.add_argument("--index", help="Search only a specific index name")
    parser.add_argument("--refresh", action="store_true", help="Rebuild the vector cache")
    parser.add_argument("--threshold", type=float, default=0.6, help="Similarity threshold (0.0-1.0)")
    parser.add_argument("-h", "--help", action="store_true", help="Show help")
    args = parser.parse_args()

    if args.help or (not args.query and not args.refresh):
        print_help()
        sys.exit(0)

    vault_cache = os.path.join(args.vault_path, ".smart-env", "scripts", "vault_cache.npz")
    
    if args.refresh:
        os.makedirs(os.path.dirname(vault_cache), exist_ok=True)
        refresh_cache(args.vault_path, vault_cache)

    if args.query:
        # Join list into string if it's multiple words
        query_text = " ".join(args.query) if isinstance(args.query, list) else args.query
        
        # Start by finding all available indices
        indices = [("vault", vault_cache)]
        if os.path.exists(CENTRAL_INDEX_STORE):
            for f in os.listdir(CENTRAL_INDEX_STORE):
                if f.endswith(".npz"):
                    name = f.replace(".npz", "")
                    indices.append((name, os.path.join(CENTRAL_INDEX_STORE, f)))
        
        # Filter by requested index if specified
        if args.index:
            indices = [idx for idx in indices if idx[0] == args.index]

        # Try daemon
        results = try_daemon_search(query_text, scope=args.scope, index=args.index, threshold=args.threshold)
        
        if results is None:
            # Slow fallback
            results = search_indexed_files(query_text, indices, scope=args.scope, threshold=args.threshold)
            
        print(json.dumps(results, indent=2))
