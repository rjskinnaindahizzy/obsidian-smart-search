"""
Shared constants, utilities, and configuration for obsidian-smart-search.

This module centralizes code that was previously duplicated across
client.py, daemon.py, and indexer.py.
"""

import os
import json
import socket
import numpy as np

# ── Network ──────────────────────────────────────────────────────────
DAEMON_HOST = '127.0.0.1'
DAEMON_PORT = 5555

# ── Paths ────────────────────────────────────────────────────────────
CENTRAL_INDEX_STORE = os.environ.get(
    "SMART_SEARCH_INDICES",
    os.path.expanduser("~/.smart-search/indices"),
)
DEFAULT_INDEX = os.environ.get("SMART_SEARCH_DEFAULT_INDEX", None)

# ── Model ────────────────────────────────────────────────────────────
MODEL_NAME = "TaylorAI/bge-micro-v2"

# ── Search defaults ──────────────────────────────────────────────────
DEFAULT_THRESHOLD = 0.45
MAX_QUERY_LENGTH = 10_000
MAX_FILE_SIZE = 1_048_576  # 1 MB

# ── Directory scanning ───────────────────────────────────────────────
SKIP_DIRS = frozenset({
    '.venv', '.git', 'node_modules', '__pycache__',
    '.smart-env', 'dist', 'build', '.next', '.cache',
    '.obsidian', '.trash',
})

INDEXABLE_EXTENSIONS = ('.md', '.txt', '.py', '.js', '.ts', '.html', '.css')

# ── Chunking ────────────────────────────────────────────────────────
CHUNK_SIZE = 2000       # characters per chunk (~400-500 tokens)
CHUNK_OVERLAP = 200     # overlap between consecutive chunks


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks, breaking at paragraph boundaries."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Try to break at a paragraph boundary (double newline)
        if end < text_len:
            break_at = text.rfind('\n\n', start + chunk_size // 2, end)
            if break_at > start:
                end = break_at

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break
        start = end - overlap

    return chunks if chunks else [text]


# ── Singleton model cache ────────────────────────────────────────────
_cached_model = None


def get_model():
    """Return the SentenceTransformer model, loading it once and caching."""
    global _cached_model
    if _cached_model is not None:
        return _cached_model
    from sentence_transformers import SentenceTransformer
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    _cached_model = SentenceTransformer(MODEL_NAME, local_files_only=True)
    return _cached_model


def cosine_similarity(query_vec, target_vecs):
    """Cosine similarity between a query vector and a matrix of target vectors.

    Handles zero-norm vectors gracefully by clamping norms to a small epsilon.
    """
    dot_product = np.dot(target_vecs, query_vec)
    norms = np.linalg.norm(target_vecs, axis=1) * np.linalg.norm(query_vec)
    norms = np.maximum(norms, 1e-10)
    return dot_product / norms


def hybrid_boost(path, query_words):
    """Unified hybrid-mode score boost based on keyword matches in *path*.

    Returns a boost value (0.0 – 0.5) to add to the semantic score.

    Boosting rules (applied in priority order):
      - Exact filename match (with or without extension): +0.4
      - Partial filename match:                          +0.2
      - Path-only match:                                 +0.1

    Multiple matching words do NOT stack beyond the single highest filename
    boost, but a path boost can add on top of a filename boost (max total +0.5).
    """
    if not query_words:
        return 0.0

    path_lower = path.lower()
    filename = os.path.basename(path).lower()

    filename_boost = 0.0
    path_boost = 0.0

    for word in query_words:
        if word == filename or word + ".py" == filename or word + ".md" == filename:
            filename_boost = max(filename_boost, 0.4)
        elif word in filename:
            filename_boost = max(filename_boost, 0.2)
        elif word in path_lower:
            path_boost = max(path_boost, 0.1)

    return filename_boost + path_boost


def try_daemon_reload():
    """Send a 'reload' command to the daemon. Returns True on success."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            s.connect((DAEMON_HOST, DAEMON_PORT))
            s.sendall(json.dumps({"command": "reload"}).encode('utf-8'))
            return True
    except (ConnectionRefusedError, TimeoutError, OSError):
        return False


def try_daemon_stop():
    """Send a 'stop' command to the daemon. Returns True on success."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.connect((DAEMON_HOST, DAEMON_PORT))
            s.sendall(json.dumps({"command": "stop"}).encode('utf-8'))
            resp = s.recv(4096).decode('utf-8', errors='replace')
            return "ok" in resp
    except (ConnectionRefusedError, TimeoutError, OSError):
        return False
