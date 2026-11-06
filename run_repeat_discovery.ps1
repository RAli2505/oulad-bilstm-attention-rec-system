param([switch]$Resume, [switch]$PreflightOnly, [switch]$ImprovedRanking)
$ErrorActionPreference = 'Stop'
# RTK requires UTF-8 bytes; Windows PowerShell otherwise uses the OEM code page.
$experimentUtf8 = New-Object System.Text.UTF8Encoding($false)
[Console]::InputEncoding = $experimentUtf8
[Console]::OutputEncoding = $experimentUtf8
$OutputEncoding = $experimentUtf8
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$experimentArgs = @(
    '-3.12', '-X', 'utf8', '-u', (Join-Path $PSScriptRoot 'run_repeat_discovery_experiment.py'),
    '--agent-run', (Join-Path $PSScriptRoot 'runs/full_agent_20261102T190210371601Z'),
    '--windows', '7', '14', '28', '56',
    '--seeds', '42', '13', '7', '23', '101',
    '--fractions', '0', '0.25', '0.5', '0.75', '1',
    '--ks', '5', '10', '20', '--bootstrap', '1000', '--threads', '6'
)
Write-Output ((Get-Date).ToString('HH:mm') + ' Python 3.12 UTF-8; experiment folder: ' + $PSScriptRoot)
$runName = 'repeat_discovery_balance_v1'
if ($ImprovedRanking) {
    $runName = 'repeat_discovery_rankmix_v2'
    $experimentArgs += '--improved-ranking'
}
if ($PreflightOnly) {
    $experimentArgs += '--preflight-only'
} elseif ($Resume) {
    $experimentArgs += @('--resume', (Join-Path $PSScriptRoot ('runs/' + $runName)))
} else {
    $experimentArgs += @('--output', (Join-Path $PSScriptRoot ('runs/' + $runName)))
}
& rtk proxy py @experimentArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
