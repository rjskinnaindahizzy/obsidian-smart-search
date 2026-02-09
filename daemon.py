import os
import json
import signal
import socket
import sys

import numpy as np

from shared import (
    DAEMON_HOST,
    DAEMON_PORT,
    CENTRAL_INDEX_STORE,
    DEFAULT_THRESHOLD,
    MAX_QUERY_LENGTH,
    MODEL_NAME,
    cosine_similarity,
    hybrid_boost,
)


class SearchDaemon:
    def __init__(self, vault_path):
        self.vault_path = vault_path
        self.vault_cache = os.path.join(vault_path, ".smart-env", "scripts", "vault_cache.npz")
        self.indices = {}  # name -> {paths, vectors}
        self._running = True

        print(f"Loading weights for {MODEL_NAME}...")
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(MODEL_NAME, local_files_only=True)

        self.reload_all_indices()

    def reload_all_indices(self):
        print("Refreshing all indices...")
        new_indices = {}

        # 1. Load vault index
        if os.path.exists(self.vault_cache):
            try:
                data = np.load(self.vault_cache, allow_pickle=False)
                new_indices["vault"] = {"paths": data["paths"], "vectors": data["vectors"]}
                print(f"Loaded 'vault' ({len(data['paths'])} vectors)")
            except Exception as e:
                print(f"Error loading vault cache: {e}")

        # 2. Load central store indices
        if os.path.exists(CENTRAL_INDEX_STORE):
            for f in os.listdir(CENTRAL_INDEX_STORE):
                if f.endswith(".npz"):
                    name = f.replace(".npz", "")
                    path = os.path.join(CENTRAL_INDEX_STORE, f)
                    try:
                        data = np.load(path, allow_pickle=False)
                        new_indices[name] = {"paths": data["paths"], "vectors": data["vectors"]}
                        print(f"Loaded '{name}' ({len(data['paths'])} vectors)")
                    except Exception as e:
                        print(f"Error loading {name}: {e}")

        self.indices = new_indices
        print("Ready.")

    def handle_search(self, query, top_k=20, threshold=DEFAULT_THRESHOLD,
                      scope=None, target_index=None, hybrid=False):
        query_vec = self.model.encode(query)
        all_results = []

        # For hybrid mode, pre-compute query words
        query_words = query.lower().split() if hybrid else []

        # Filter indices to search (--index all or absent → search everything)
        to_search = self.indices.items()
        if target_index and target_index.lower() != "all" and target_index in self.indices:
            to_search = [(target_index, self.indices[target_index])]

        for label, data in to_search:
            paths = data["paths"]
            vectors = data["vectors"]
            scores = cosine_similarity(query_vec, vectors)

            for i, score in enumerate(scores):
                path = str(paths[i])
                effective_score = float(score)

                if hybrid:
                    effective_score = min(1.0, effective_score + hybrid_boost(path, query_words))

                if effective_score >= threshold:
                    if not scope or scope.lower() in path.lower():
                        all_results.append({"path": path, "score": effective_score, "index": label})

        all_results.sort(key=lambda x: x["score"], reverse=True)

        seen = set()
        unique = []
        for r in all_results:
            if r["path"] not in seen:
                unique.append(r)
                seen.add(r["path"])
            if len(unique) >= top_k:
                break

        return unique

    # ── Graceful shutdown ────────────────────────────────────────────
    def _handle_signal(self, signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
        self._running = False

    def run(self):
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((DAEMON_HOST, DAEMON_PORT))
            s.listen()
            s.settimeout(1.0)  # Allow periodic check of self._running
            print(f"Search Daemon listening on {DAEMON_HOST}:{DAEMON_PORT}")

            while self._running:
                try:
                    conn, addr = s.accept()
                except socket.timeout:
                    continue  # Check self._running and loop

                with conn:
                    try:
                        raw = conn.recv(4096)
                        if not raw:
                            continue
                        data = raw.decode('utf-8', errors='replace')
                    except OSError:
                        continue

                    try:
                        request = json.loads(data)
                        cmd = request.get("command")

                        if cmd == "search":
                            query = request.get("query", "")
                            if len(query) > MAX_QUERY_LENGTH:
                                conn.sendall(json.dumps(
                                    {"error": f"Query exceeds maximum length of {MAX_QUERY_LENGTH} chars"}
                                ).encode('utf-8'))
                                continue

                            results = self.handle_search(
                                query,
                                scope=request.get("scope"),
                                target_index=request.get("index"),
                                threshold=request.get("threshold", DEFAULT_THRESHOLD),
                                hybrid=request.get("hybrid", False),
                            )
                            conn.sendall(json.dumps(results).encode('utf-8'))

                        elif cmd == "reload":
                            self.reload_all_indices()
                            conn.sendall(b'{"status": "ok"}')

                        elif cmd == "ping":
                            conn.sendall(b'{"status": "pong"}')

                        elif cmd == "stop":
                            conn.sendall(b'{"status": "ok", "message": "shutting down"}')
                            print("Received stop command, shutting down...")
                            self._running = False

                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        conn.sendall(json.dumps({"error": str(e)}).encode('utf-8'))

        print("Daemon stopped.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python daemon.py <vault_path>")
        sys.exit(1)

    daemon = SearchDaemon(sys.argv[1])
    daemon.run()
