$projectDirectory = $PSScriptRoot
$tradingAgentsCli = Join-Path $projectDirectory ".venv\Scripts\tradingagents.exe"

Set-Location -LiteralPath $projectDirectory

if (-not (Test-Path -LiteralPath $tradingAgentsCli)) {
    Write-Error "TradingAgents was not found at: $tradingAgentsCli"
    Write-Host "Recreate the project's .venv environment, then try again."
    exit 1
}

& $tradingAgentsCli
