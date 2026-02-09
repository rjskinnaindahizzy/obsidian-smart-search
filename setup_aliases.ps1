$ScriptDir = $PSScriptRoot
$ProfilePath = $PROFILE

Write-Host "Setting up Obsidian Smart Search aliases..."
Write-Host "Repo location: $ScriptDir"
Write-Host "Profile: $ProfilePath"

# 1. Handle Default Vault Path
$DefaultVault = $env:OBSIDIAN_VAULT_PATH
if (-not $DefaultVault) {
    Write-Host "`nNo default vault path detected (OBSIDIAN_VAULT_PATH is empty)." -ForegroundColor Cyan
    $InputPath = Read-Host "Please enter the absolute path to your Obsidian Vault (or press Enter to skip)"
    if ($InputPath) {
        # Check if user provided path with quotes and strip them
        $InputPath = $InputPath.Trim('"', "'")
        if (Test-Path $InputPath) {
            $DefaultVault = (Convert-Path $InputPath)
            Write-Host "Setting default vault to: $DefaultVault" -ForegroundColor Green
        } else {
            Write-Host "Warning: Path '$InputPath' not found. Skipping default set." -ForegroundColor Yellow
        }
    }
}

# 2. Build Profile Content
$ProfileContent = "`n# Obsidian Smart Search Configuration`n"
if ($DefaultVault) {
    $ProfileContent += "`$env:OBSIDIAN_VAULT_PATH = `"$DefaultVault`"`n"
    # Note: SMART_SEARCH_DEFAULT_INDEX is NOT set by default so that external
    # indices (created via obs-index) are also searchable.  To restrict searches
    # to the vault only, uncomment the line below or set the variable manually.
    # $ProfileContent += "`$env:SMART_SEARCH_DEFAULT_INDEX = `"vault`"  # Only search vault by default`n"
}

$Aliases = @"

# Obsidian Smart Search Aliases
# Added by setup_aliases.ps1
`$ObsSearchPath = "$ScriptDir"

function obs-search-server {
    param([string]`$VaultPath)
    # Support unquoted paths with spaces
    if (-not `$VaultPath -and `$args) { `$VaultPath = `$args -join " " }
    if (-not `$VaultPath) { `$VaultPath = `$env:OBSIDIAN_VAULT_PATH }
    
    if (-not `$VaultPath) {
        Write-Host "Error: No vault path provided and OBSIDIAN_VAULT_PATH is not set." -ForegroundColor Red
        Write-Host "Usage: obs-search-server <path-to-vault>"
        return
    }
    `$Target = Convert-Path `$VaultPath
    Start-Job -Name "ObsidianSearchDaemon" -ScriptBlock {
        param(`$ScriptPath, `$VaultPath)
        Set-Location `$VaultPath
        uv run --with numpy --with sentence-transformers `$ScriptPath `$VaultPath
    } -ArgumentList "`$ObsSearchPath\daemon.py", `$Target
    Write-Host "Search Daemon started in background for: " `$Target
}

function obs-search-stop {
    # Try socket-based stop first (works regardless of how daemon was started)
    uv run --with numpy --with sentence-transformers "`$ObsSearchPath\client.py" --stop
    # Also clean up PowerShell background job if one exists
    Get-Job -Name "ObsidianSearchDaemon" -ErrorAction SilentlyContinue | Stop-Job
    Get-Job -Name "ObsidianSearchDaemon" -ErrorAction SilentlyContinue | Remove-Job
}

function idx-external {
    param([string]`$Path, [string]`$Name)
    # Support unquoted path for first arg
    if (-not `$Path -and `$args) { 
        # This one is trickier if name is also provided, 
        # but usually people use quotes if using multiple args.
        `$Path = `$args[0]
        `$Name = `$args[1]
    }
    uv run --with numpy --with sentence-transformers "`$ObsSearchPath\indexer.py" `$Path `$Name
}
# Alias for consistency with new docs
if (-not (Get-Alias obs-index -ErrorAction SilentlyContinue)) {
    Set-Alias -Name obs-index -Value idx-external
}

function obs-search {
    # client.py already handles --vault_path correctly. 
    # This wrapper passes all args.
    uv run --with numpy --with sentence-transformers "`$ObsSearchPath\client.py" `$args
}
"@

$ProfileContent += $Aliases

# 3. Write to Profile
if (Test-Path $ProfilePath) {
    # Check if we already have these markers to avoid duplicates
    $CurrentProfile = Get-Content $ProfilePath -Raw
    if ($CurrentProfile -match "Obsidian Smart Search Aliases") {
        Write-Host "Aliases already exist in profile. Updating is manual for now to avoid mess." -ForegroundColor Yellow
    } else {
        Add-Content -Path $ProfilePath -Value $ProfileContent
        Write-Host "Aliases appended to your PowerShell profile."
        Write-Host "Please run '. `$PROFILE' to reload your profile."
    }
} else {
    Write-Host "Profile not found at $ProfilePath. Please create it and add the following:"
    Write-Host $ProfileContent
}
