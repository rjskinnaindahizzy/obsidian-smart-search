$ScriptDir = $PSScriptRoot
$ProfilePath = $PROFILE

Write-Host "Setting up Obsidian Smart Search aliases..."
Write-Host "Repo location: $ScriptDir"
Write-Host "Profile: $ProfilePath"

$Aliases = @"

# Obsidian Smart Search Aliases
# Added by setup_aliases.ps1
function obs-search-server {
    Start-Job -Name "ObsidianSearchDaemon" -ScriptBlock {
        uv run --with numpy --with sentence-transformers "$ScriptDir\daemon.py" "e:\Obsidian Vault"
    }
    Write-Host "Search Daemon started in background."
}

function obs-search-stop {
    Get-Job -Name "ObsidianSearchDaemon" -ErrorAction SilentlyContinue | Stop-Job
    Get-Job -Name "ObsidianSearchDaemon" -ErrorAction SilentlyContinue | Remove-Job
    Write-Host "Search Booster stopped."
}

function idx-external {
    param(
        [Parameter(Mandatory=$true, Position=0)] [string]$Path,
        [Parameter(Mandatory=$true, Position=1)] [string]$Name
    )
    uv run --with numpy --with sentence-transformers "$ScriptDir\indexer.py" $Path $Name
}
# Alias for consistency with new docs
Set-Alias -Name obs-index -Value idx-external

function obs-search {
    # Pass all arguments to the python script
    uv run --with numpy --with sentence-transformers "$ScriptDir\client.py" $args
}
"@

if (Test-Path $ProfilePath) {
    Add-Content -Path $ProfilePath -Value $Aliases
    Write-Host "Aliases appended to your PowerShell profile."
    Write-Host "Please run '. $ProfilePath' to reload your profile."
} else {
    Write-Host "Profile not found at $ProfilePath. Please create it and add the following:"
    Write-Host $Aliases
}
