import os
import json
import sys
import argparse
import socket
import subprocess
import time

import numpy as np

from shared import (
    DAEMON_HOST,
    DAEMON_PORT,
    CENTRAL_INDEX_STORE,
    DEFAULT_INDEX,
    DEFAULT_THRESHOLD,
    INDEXABLE_EXTENSIONS,
    MAX_QUERY_LENGTH,
    MAX_FILE_SIZE,
    SKIP_DIRS,
    chunk_text,
    get_model,
    cosine_similarity,
    hybrid_boost,
    try_daemon_reload,
    try_daemon_stop,
)


def start_daemon(vault_path):
    """Start the Search Booster daemon in the background."""
    # Validate vault_path before spawning a subprocess
    vault_path = os.path.realpath(vault_path)
    if not os.path.isdir(vault_path):
        print(f"Error: vault path is not an existing directory: {vault_path}", file=sys.stderr)
        return False

    script_dir = os.path.dirname(os.path.abspath(__file__))
    daemon_script = os.path.join(script_dir, "daemon.py")

    if not os.path.exists(daemon_script):
        print(f"Error: daemon.py not found at {daemon_script}", file=sys.stderr)
        return False

    try:
        print(f"Starting Search Booster for {vault_path}...", file=sys.stderr)
        subprocess.Popen(
            [sys.executable, "-m", "uv", "run", "--with", "numpy", "--with", "sentence-transformers",
             daemon_script, vault_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True
    except Exception as e:
        print(f"Failed to start daemon: {e}", file=sys.stderr)
        return False


def _send_daemon_request(request, timeout=0.2):
    """Send a JSON request to the daemon and return the parsed response.

    Returns None on any connection or protocol error.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect((DAEMON_HOST, DAEMON_PORT))
        s.sendall(json.dumps(request).encode('utf-8'))
        response = s.recv(131072).decode('utf-8', errors='replace')
        return json.loads(response)


def try_daemon_search(query, scope=None, index=None, threshold=DEFAULT_THRESHOLD,
                      vault_path=None, auto_start=True, hybrid=False):
    """Try to search using the daemon. Auto-start if not running."""
    request = {
        "command": "search",
        "query": query,
        "scope": scope,
        "index": index,
        "threshold": threshold,
        "hybrid": hybrid,
    }

    try:
        return _send_daemon_request(request)
    except (ConnectionRefusedError, TimeoutError, OSError):
        pass  # Daemon not running — try auto-start below
    except (json.JSONDecodeError, ValueError):
        return None  # Bad response from daemon

    if not (auto_start and vault_path):
        return None

    if not start_daemon(vault_path):
        return None

    # Retry with exponential backoff: 0.5s, 1.0s, 2.0s
    for delay in (0.5, 1.0, 2.0):
        time.sleep(delay)
        try:
            result = _send_daemon_request(request, timeout=1.0)
            print("Search Booster started successfully.", file=sys.stderr)
            return result
        except (ConnectionRefusedError, TimeoutError, OSError):
            continue
        except (json.JSONDecodeError, ValueError):
            return None

    return None


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
                    if not line:
                        continue
                    if line.endswith(","):
                        line = line[:-1]
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
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue

    np.savez_compressed(cache_path, paths=np.array(paths), vectors=np.array(vectors))
    print(f"Cache saved to {cache_path} ({len(paths)} vectors)")

    if try_daemon_reload():
        print("Notified Search Booster to reload.")


def search_indexed_files(query, indices, top_k=20, threshold=DEFAULT_THRESHOLD,
                         scope=None, hybrid=False):
    """Search indexed files. If hybrid=True, boost scores for keyword matches."""
    all_results = []
    model = None
    query_vec = None
    query_words = query.lower().split() if hybrid else []

    for label, cache_path in indices:
        if not os.path.exists(cache_path):
            continue

        try:
            data = np.load(cache_path, allow_pickle=False)
            paths = data["paths"]
            vectors = data["vectors"]

            if model is None:
                model = get_model()
                query_vec = model.encode(query)

            scores = cosine_similarity(query_vec, vectors)

            for i, score in enumerate(scores):
                path = str(paths[i])
                effective_score = float(score)

                if hybrid:
                    effective_score = min(1.0, effective_score + hybrid_boost(path, query_words))

                if effective_score >= threshold:
                    if not scope or scope.lower() in path.lower():
                        all_results.append({"path": path, "score": effective_score, "index": label})
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
        if len(unique) >= top_k:
            break

    return unique


def get_cache_name_for_path(dir_path):
    """Generate a safe cache filename from a directory path."""
    normalized = os.path.normpath(os.path.abspath(dir_path))
    safe_name = normalized.replace("\\", "_").replace("/", "_").replace(":", "").strip("_")
    return f"autoscan_{safe_name[:100]}"


def search_unindexed_directory(query, dir_path, top_k=20, threshold=DEFAULT_THRESHOLD):
    """Scan and index a directory. Caches results for future searches."""
    cache_name = get_cache_name_for_path(dir_path)
    cache_path = os.path.join(CENTRAL_INDEX_STORE, f"{cache_name}.npz")

    if os.path.exists(cache_path):
        print(f"Using cached index: {cache_name}", file=sys.stderr)
        try:
            data = np.load(cache_path, allow_pickle=False)
            paths = data["paths"]
            vectors = data["vectors"]

            model = get_model()
            query_vec = model.encode(query)
            scores = cosine_similarity(query_vec, vectors)

            results = []
            for i, score in enumerate(scores):
                if score >= threshold:
                    results.append({"path": str(paths[i]), "score": float(score), "index": cache_name})
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
        except (OSError, ValueError) as e:
            print(f"Cache read failed, rescanning: {e}", file=sys.stderr)

    # No cache — scan, chunk, and embed
    print(f"Scanning unindexed directory: {dir_path}...", file=sys.stderr)
    file_paths = []
    file_texts = []

    for root, dirs, files in os.walk(dir_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for file in files:
            if file.endswith(INDEXABLE_EXTENSIONS):
                file_path = os.path.join(root, file)
                try:
                    if os.path.getsize(file_path) > MAX_FILE_SIZE:
                        print(f"Skipping large file ({os.path.getsize(file_path)} bytes): {file_path}", file=sys.stderr)
                        continue
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                        if content.strip():
                            file_paths.append(file_path)
                            file_texts.append(content)
                except OSError:
                    continue

    if not file_paths:
        return []

    # Chunk files for sharper embeddings
    chunk_paths = []
    chunk_texts = []
    for fpath, ftext in zip(file_paths, file_texts):
        for chunk in chunk_text(ftext):
            chunk_paths.append(fpath)
            chunk_texts.append(chunk)

    print(f"Embedding {len(file_paths)} files ({len(chunk_texts)} chunks)...", file=sys.stderr)
    model = get_model()
    query_vec = model.encode(query)
    doc_vecs = model.encode(chunk_texts, show_progress_bar=True)

    # Cache the chunked embeddings for future use
    os.makedirs(CENTRAL_INDEX_STORE, exist_ok=True)
    np.savez_compressed(cache_path, paths=np.array(chunk_paths), vectors=doc_vecs)
    print(f"Cached index saved: {cache_path}", file=sys.stderr)

    scores = cosine_similarity(query_vec, doc_vecs)

    results = []
    for i, score in enumerate(scores):
        if score >= threshold:
            results.append({"path": str(chunk_paths[i]), "score": float(score), "index": cache_name})

    # Deduplicate: keep highest-scoring chunk per file
    results.sort(key=lambda x: x["score"], reverse=True)
    seen = set()
    unique = []
    for r in results:
        if r["path"] not in seen:
            unique.append(r)
            seen.add(r["path"])
        if len(unique) >= top_k:
            break

    return unique


def remove_index(index_name, vault_path=None):
    """Remove an index by name."""
    success = False

    if index_name == "vault":
        if not vault_path:
            print("Error: Removing vault index requires --vault_path or OBSIDIAN_VAULT_PATH.", file=sys.stderr)
            return False
        vault_cache = os.path.join(vault_path, ".smart-env", "scripts", "vault_cache.npz")
        if os.path.exists(vault_cache):
            try:
                os.remove(vault_cache)
                print(f"Removed vault index cache: {vault_cache}")
                success = True
            except OSError as e:
                print(f"Error removing vault index: {e}", file=sys.stderr)
        else:
            print("Vault index not found.")

    elif os.path.exists(CENTRAL_INDEX_STORE):
        path = os.path.join(CENTRAL_INDEX_STORE, f"{index_name}.npz")
        if os.path.exists(path):
            try:
                os.remove(path)
                print(f"Removed index '{index_name}' from store ({path}).")
                success = True
            except OSError as e:
                print(f"Error removing index '{index_name}': {e}", file=sys.stderr)
        else:
            print(f"Index '{index_name}' not found in store.")
    else:
        print(f"Index store {CENTRAL_INDEX_STORE} does not exist.")

    if success:
        if try_daemon_reload():
            print("Notified Search Booster to reload indices.")
        return True
    return False


def list_indices(vault_cache):
    """List all available indices with metadata."""
    from datetime import datetime

    indices = []

    # Check vault cache
    if os.path.exists(vault_cache):
        try:
            data = np.load(vault_cache, allow_pickle=False)
            stat = os.stat(vault_cache)
            total = len(data["paths"])
            unique = len(set(str(p) for p in data["paths"]))
            indices.append({
                "name": "vault",
                "path": vault_cache,
                "files": unique,
                "chunks": total,
                "size_kb": stat.st_size // 1024,
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            })
        except (OSError, ValueError):
            pass

    # Check central store
    if os.path.exists(CENTRAL_INDEX_STORE):
        for f in os.listdir(CENTRAL_INDEX_STORE):
            if f.endswith(".npz"):
                path = os.path.join(CENTRAL_INDEX_STORE, f)
                name = f.replace(".npz", "")
                try:
                    data = np.load(path, allow_pickle=False)
                    stat = os.stat(path)
                    total = len(data["paths"])
                    unique = len(set(str(p) for p in data["paths"]))
                    indices.append({
                        "name": name,
                        "path": path,
                        "files": unique,
                        "chunks": total,
                        "size_kb": stat.st_size // 1024,
                        "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                    })
                except (OSError, ValueError):
                    continue

    if not indices:
        print("No indices found.")
        print(f"Index store: {CENTRAL_INDEX_STORE}")
        return

    print(f"{'Name':<30} {'Files':>8} {'Chunks':>8} {'Size':>10} {'Modified':<16}")
    print("-" * 80)
    for idx in indices:
        print(f"{idx['name']:<30} {idx['files']:>8} {idx['chunks']:>8} {idx['size_kb']:>7} KB  {idx['modified']:<16}")
    print(f"\nIndex store: {CENTRAL_INDEX_STORE}")


def print_help():
    help_text = f"""
Obsidian Semantic Search (Global)
================================

Usage:
  obs-search "<query>" [options]

Options:
  --scope <folder>   Filter results to paths containing this string.
                     Note: If folder isn't indexed, a one-off slow scan will run.
  --index <name>     Search only a specific index (e.g., 'vault', 'documents').
                     Use '--index all' to search all indices (overrides default).
  --list             List all available indices.
  --refresh          Rebuild the Obsidian vault cache.
  --remove <name>    Remove an index by name.
  --threshold <val>  Similarity threshold (0.0-1.0, default {DEFAULT_THRESHOLD}).
  --hybrid           Combine semantic + keyword matching (default: on).
  --no-hybrid        Disable hybrid mode (pure semantic search).
  --stop             Stop the background Search Booster daemon.
  -h, --help         Show this help menu.

Booster Commands:
  obs-search-server  Start background booster (sub-second search).
  obs-search-stop    Stop background booster.

External Indexing:
  obs-index <path> <name>   Index a new external directory (recommended for speed).
    """
    print(help_text.strip())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("query", nargs="*", help="Search query")
    parser.add_argument("--vault_path", default=os.environ.get("OBSIDIAN_VAULT_PATH"), help="Path to Obsidian vault")
    parser.add_argument("--scope", help="Optional folder/path substring to scope search")
    parser.add_argument("--index", help="Search only a specific index name")
    parser.add_argument("--list", action="store_true", help="List all available indices")
    parser.add_argument("--refresh", action="store_true", help="Rebuild the vector cache")
    parser.add_argument("--remove", help="Remove an index by name")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Similarity threshold (0.0-1.0)")
    parser.add_argument("--hybrid", action="store_true", default=True, help="Combine semantic + keyword matching (default: on)")
    parser.add_argument("--no-hybrid", dest="hybrid", action="store_false", help="Disable hybrid mode (pure semantic search)")
    parser.add_argument("--stop", action="store_true", help="Stop the Search Booster daemon")
    parser.add_argument("-h", "--help", action="store_true", help="Show help")
    args = parser.parse_args()

    # Handle --stop before anything else (doesn't require vault_path)
    if args.stop:
        if try_daemon_stop():
            print("Search Booster daemon stopped.")
        else:
            print("Could not connect to daemon (is it running?).", file=sys.stderr)
        sys.exit(0)

    if args.help or (not args.query and not args.refresh and not args.list and not args.remove):
        print_help()
        sys.exit(0)

    if not args.vault_path:
        print("Error: No vault path provided. Use --vault_path or set OBSIDIAN_VAULT_PATH environment variable.", file=sys.stderr)
        sys.exit(1)

    vault_cache = os.path.join(args.vault_path, ".smart-env", "scripts", "vault_cache.npz")

    if args.list:
        list_indices(vault_cache)
        sys.exit(0)

    if args.remove:
        remove_index(args.remove, args.vault_path)
        sys.exit(0)

    if args.refresh:
        os.makedirs(os.path.dirname(vault_cache), exist_ok=True)
        refresh_cache(args.vault_path, vault_cache)

    if args.query:
        query_text = " ".join(args.query) if isinstance(args.query, list) else args.query

        # Validate query length
        if len(query_text) > MAX_QUERY_LENGTH:
            print(f"Error: Query exceeds maximum length of {MAX_QUERY_LENGTH} characters.", file=sys.stderr)
            sys.exit(1)

        # Build index list
        indices = [("vault", vault_cache)]
        if os.path.exists(CENTRAL_INDEX_STORE):
            for f in os.listdir(CENTRAL_INDEX_STORE):
                if f.endswith(".npz"):
                    name = f.replace(".npz", "")
                    indices.append((name, os.path.join(CENTRAL_INDEX_STORE, f)))

        # Filter by requested index if specified (or use default)
        effective_index = args.index or DEFAULT_INDEX
        if effective_index and effective_index.lower() != "all":
            filtered = [idx for idx in indices if idx[0] == effective_index]
            if not filtered and indices:
                available = [name for name, _ in indices]
                print(f"Warning: No index named '{effective_index}' found. "
                      f"Available: {', '.join(available)}. Searching all indices.",
                      file=sys.stderr)
            else:
                indices = filtered

        # Try daemon (with auto-start if not running)
        results = try_daemon_search(
            query_text, scope=args.scope, index=effective_index,
            threshold=args.threshold, vault_path=args.vault_path, hybrid=args.hybrid,
        )

        if results is None or (results == [] and indices):
            # Daemon unavailable or returned nothing — try local search
            local_results = search_indexed_files(
                query_text, indices, scope=args.scope,
                threshold=args.threshold, hybrid=args.hybrid,
            )
            if local_results:
                results = local_results

        if not results and args.scope and os.path.isdir(args.scope):
            results = search_unindexed_directory(query_text, args.scope, threshold=args.threshold)

        print(json.dumps(results, indent=2))
