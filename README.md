# Obsidian Smart Search

A powerful, semantic search engine for your Obsidian vault and external directories. Supports sub-second "instant search" via a background daemon and path-based scoping.

## Features

- **Semantic Search**: Uses `TaylorAI/bge-micro-v2` for high-quality, local embeddings.
- **Instant Search Daemon**: A background service that keeps the model in memory for sub-second responses.
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

3. **Setup PowerShell Aliases (Optional but Recommended):**
    Run the included setup script to add `obs-search`, `obs-index`, and `obs-search-server` to your PowerShell profile.

    ```powershell
    .\setup_aliases.ps1
    ```

## Usage

### 1. Basic Search

Search your Obsidian vault:

```bash
obs-search "how to handle prompt injection"
```

### 2. Global Search & Scoping

Index an external directory (e.g., a coding project):

```bash
obs-index "C:\Users\username\Documents\Projects" projects
```

Search everything (Vault + Projects):

```bash
obs-search "invoice generation"
```

Search ONLY within the "Projects" path:

```bash
obs-search "invoice generation" --scope "Projects"
```

Search ONLY the "projects" index (ignore vault):

```bash
obs-search "invoice generation" --index "projects"
```

### 3. Instant Search Booster

Start the daemon to cache the model in memory. Subsequent searches will be instant.

```bash
obs-search-server
```

You will see: `Search Daemon listening on 127.0.0.1:5555`

To stop it:

```bash
obs-search-stop
```

## Configuration

Indices are stored by default in `~/.smart-search/indices`. You can override this by setting the `SMART_SEARCH_INDICES` environment variable.

## Architecture

- **client.py**: The CLI tool (`obs-search`). Connects to the daemon if available, otherwise runs a one-off search (slower).
- **daemon.py**: The background server (`obs-search-server`). Maintains persistent model state and indices.
- **indexer.py**: The indexing tool (`obs-index`). Scans directories and saves `.npz` vector files.
