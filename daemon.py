import os
import json
import socket
import numpy as np
from sentence_transformers import SentenceTransformer
import sys

# Constants
PORT = 5555
HOST = '127.0.0.1'
MODEL_NAME = "TaylorAI/bge-micro-v2"
CENTRAL_INDEX_STORE = os.environ.get("SMART_SEARCH_INDICES", os.path.expanduser("~/.smart-search/indices"))

def cosine_similarity(query_vec, target_vecs):
    dot_product = np.dot(target_vecs, query_vec)
    norms = np.linalg.norm(target_vecs, axis=1) * np.linalg.norm(query_vec)
    return dot_product / norms

class SearchDaemon:
    def __init__(self, vault_path):
        self.vault_path = vault_path
        self.vault_cache = os.path.join(vault_path, ".smart-env", "scripts", "vault_cache.npz")
        self.indices = {} # name -> {paths, vectors}
        
        print(f"Loading weights for {MODEL_NAME}...")
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        self.model = SentenceTransformer(MODEL_NAME, local_files_only=True)
        
        self.reload_all_indices()
        
    def reload_all_indices(self):
        print(f"Refreshing all indices...")
        new_indices = {}
        
        # 1. Load vault index
        if os.path.exists(self.vault_cache):
            try:
                data = np.load(self.vault_cache)
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
                        data = np.load(path)
                        new_indices[name] = {"paths": data["paths"], "vectors": data["vectors"]}
                        print(f"Loaded '{name}' ({len(data['paths'])} vectors)")
                    except Exception as e:
                        print(f"Error loading {name}: {e}")
        
        self.indices = new_indices
        print("Ready.")

    def handle_search(self, query, top_k=20, threshold=0.6, scope=None, target_index=None):
        query_vec = self.model.encode(query)
        all_results = []
        
        # Filter indices to search
        to_search = self.indices.items()
        if target_index and target_index in self.indices:
            to_search = [(target_index, self.indices[target_index])]
            
        for label, data in to_search:
            paths = data["paths"]
            vectors = data["vectors"]
            scores = cosine_similarity(query_vec, vectors)
            
            for i, score in enumerate(scores):
                path = str(paths[i])
                if score >= threshold:
                    if not scope or scope.lower() in path.lower():
                        all_results.append({"path": path, "score": float(score), "index": label})
                        
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        seen = set()
        unique = []
        for r in all_results:
            if r["path"] not in seen:
                unique.append(r)
                seen.add(r["path"])
            if len(unique) >= top_k: break
            
        return unique

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            s.listen()
            print(f"Search Daemon listening on {HOST}:{PORT}")
            
            while True:
                conn, addr = s.accept()
                with conn:
                    data = conn.recv(4096).decode('utf-8')
                    if not data: continue
                    
                    try:
                        request = json.loads(data)
                        cmd = request.get("command")
                        if cmd == "search":
                            results = self.handle_search(
                                request["query"],
                                scope=request.get("scope"),
                                target_index=request.get("index"),
                                threshold=request.get("threshold", 0.6)
                            )
                            conn.sendall(json.dumps(results).encode('utf-8'))
                        elif cmd == "reload":
                            self.reload_all_indices()
                            conn.sendall(b'{"status": "ok"}')
                        elif cmd == "ping":
                            conn.sendall(b'{"status": "pong"}')
                    except Exception as e:
                        conn.sendall(json.dumps({"error": str(e)}).encode('utf-8'))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python search_daemon.py <vault_path>")
        sys.exit(1)
        
    daemon = SearchDaemon(sys.argv[1])
    daemon.run()
