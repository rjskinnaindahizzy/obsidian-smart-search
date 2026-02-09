# Obsidian Smart Search

A powerful, semantic search engine for your Obsidian vault and external directories. Supports sub-second "instant search" via a background daemon and path-based scoping.

## Features

- **Semantic Search**: Uses `TaylorAI/bge-micro-v2` for high-quality, local embeddings.
- **Chunked Indexing**: Files are split into overlapping ~2000-character chunks for sharper, higher-quality embeddings. The best-matching chunk determines each file's score.
- **Instant Search Daemon**: A background service that keeps the model in memory for sub-second responses.
- **Hybrid Search**: Combines semantic search with keyword path-matching (on by default). Use `--no-hybrid` for pure semantic mode.
- **Global Indexing**: Index folders *outside* your vault and search them alongside your notes.
- **Path Scoping**: Filter search results to specific folders or sub-paths (`--scope`).
- **Multi-Index**: Manage and target separate indices (`--index`).

## Installation

1. **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd obsidian-smart-search
    ```

2. **Install dependencies:**
    It is recommended to use `uv` for fast dependency management.

    ```bash
    pip install -r requirements.txt
    # OR with uv
    uv pip install -r requirements.txt
    ```

3. **Download the embedding model (first time only):**

    ```bash
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('TaylorAI/bge-micro-v2')"
    ```

    This downloads the model (~22 MB) so it's available offline. All subsequent runs use the local cache.

4. **Setup PowerShell Aliases (Optional but Recommended):**
    Run the included setup script to add `obs-search`, `obs-index`, and `obs-search-server` to your PowerShell profile.

    ```powershell
    .\setup_aliases.ps1
    ```

## Configuration

The project uses environment variables to avoid hardcoded paths.

- `OBSIDIAN_VAULT_PATH`: The path to your Obsidian vault.
- `SMART_SEARCH_DEFAULT_INDEX`: (Optional) Restrict searches to a single index by name (e.g., `vault`). When unset (default), all indices are searched. Use `--index all` at search time to override.
- `SMART_SEARCH_INDICES`: (Optional) Path to store indices (defaults to `~/.smart-search/indices`).

You can set these in your PowerShell profile:

```powershell
$env:OBSIDIAN_VAULT_PATH = "C:\Path\To\Your\Vault"
# $env:SMART_SEARCH_DEFAULT_INDEX = "vault"  # Uncomment to only search vault by default
```

## Usage

### 1. Setup Aliases

Run `setup_aliases.ps1` to add the search commands to your PowerShell profile.
**Note**: The script will prompt you for your default Obsidian Vault path if it's not already set.

```powershell
.\setup_aliases.ps1
```

### 2. Basic Search

Search your Obsidian vault:

```bash
obs-search "how to handle prompt injection"
```

### 3. Global Search & Scoping

Index an external directory (e.g., a coding project):

```bash
obs-index "C:\Users\username\Documents\Projects" projects
```

Search everything (Vault + Projects):

```bash
obs-search "invoice generation"
```

> [!TIP]
> **Slow Fallback**: If you use `--scope` on a folder that hasn't been indexed (e.g., `obs-search "query" --scope C:\Downloads`), the tool will automatically perform a "live scan" and embed files on-the-fly. This is slower but useful for one-off searches.

Search ONLY within the "Projects" path (assuming it has been indexed):

```bash
obs-search "invoice generation" --scope "Projects"
```

Search ONLY the "projects" index (ignore vault):

```bash
obs-search "invoice generation" --index "projects"
```

### 4. Hybrid Search (on by default)

Hybrid mode combines semantic similarity with keyword path-matching. It's enabled by default because it significantly improves result quality with small embedding models.

```bash
obs-search "red team"              # hybrid is on by default
obs-search "red team" --no-hybrid  # pure semantic search
```

> [!TIP]
> **Hybrid mode** boosts scores when query words appear in the file path. Great for finding files when you remember part of the name but also want semantic matches.

### 5. Instant Search Booster

Start the daemon to cache the model in memory. Subsequent searches will be instant.

```bash
obs-search-server
```

You will see: `Search Daemon listening on 127.0.0.1:5555`

To stop it:

```bash
obs-search-stop
```

## Architecture

- **shared.py**: Shared constants, utilities, and configuration. Contains `cosine_similarity()`, `get_model()`, `chunk_text()`, hybrid boosting logic, and network/path constants used by all other modules.
- **client.py**: The CLI tool (`obs-search`). Connects to the daemon if available, otherwise runs a one-off search (slower). **Includes a "live scan" mode for unindexed scopes.**
- **daemon.py**: The background server (`obs-search-server`). Maintains persistent model state and indices. Supports graceful shutdown via `--stop` or SIGTERM.
- **indexer.py**: The indexing tool (`obs-index`). Scans directories, splits files into chunks, and saves `.npz` vector files.

> [!NOTE]
> **Re-indexing required after upgrade**: If you're upgrading from a previous version, re-run `obs-index` on your directories. The new chunked format produces more vectors per file, which dramatically improves search quality.
