param([switch]$CheckOnly, [switch]$Original)
$ErrorActionPreference = 'Stop'
$experimentUtf8 = New-Object System.Text.UTF8Encoding($false)
[Console]::InputEncoding = $experimentUtf8
[Console]::OutputEncoding = $experimentUtf8
$OutputEncoding = $experimentUtf8
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$runPath = Join-Path $PSScriptRoot 'runs/full_agent_20261102T190210371601Z'
$saved = Get-Content -LiteralPath (Join-Path $runPath 'manifest.json') -Encoding UTF8 -Raw | ConvertFrom-Json
if ($saved.external_test_used -or $saved.pilot_not_for_publication) { throw 'A full 2013-only source is required.' }
if ($saved.sasrec_dimension -ne 64) { throw 'CLI supports the default SASRec dimension 64 only.' }
$hashes = $saved.resume_source_code_sha256
if ($null -eq $hashes) { $hashes = $saved.source_code_sha256 }
foreach ($property in $hashes.PSObject.Properties) {
    $hashAlgorithm = [System.Security.Cryptography.SHA256]::Create()
    $hashStream = [System.IO.File]::OpenRead((Join-Path $PSScriptRoot $property.Name))
    try { $actual = [BitConverter]::ToString($hashAlgorithm.ComputeHash($hashStream)).Replace('-', '').ToLowerInvariant() }
    finally { $hashStream.Dispose(); $hashAlgorithm.Dispose() }
    if ($actual -ne $property.Value) { throw ('Resume source mismatch: ' + $property.Name) }
}
$entryPoint = Join-Path $PSScriptRoot 'run_oulad_agent_experiment.py'
$accelerationArgs = @()
if (-not $Original) {
    $entryPoint = Join-Path $PSScriptRoot 'fast_full_agent_resume.py'
    $validationPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'tmp/fast_full_agent_validation/report.json'
    $validation = Get-Content -LiteralPath $validationPath -Encoding UTF8 -Raw | ConvertFrom-Json
    if (-not $validation.exact_equal) { throw 'Acceleration has not passed exact parity validation.' }
    $accelerationArgs = @('--validation-report', $validationPath)
}
$experimentArgs = @('-3.12', '-X', 'utf8', '-u', $entryPoint)
$experimentArgs += $accelerationArgs
$experimentArgs += @(
    '--risk-run', $saved.risk_source, '--resume', $runPath, '--raw', $saved.raw_source, '--windows')
$experimentArgs += @($saved.windows | ForEach-Object { [string]$_ })
$experimentArgs += '--seeds'
$experimentArgs += @($saved.seeds | ForEach-Object { [string]$_ })
$experimentArgs += '--ks'
$experimentArgs += @($saved.ks | ForEach-Object { [string]$_ })
$experimentArgs += @('--horizon', [string]$saved.horizon, '--bootstrap', [string]$saved.bootstrap,
    '--threads', [string]$saved.threads, '--epochs', [string]$saved.epochs,
    '--patience', [string]$saved.patience, '--device', [string]$saved.device,
    '--inference-batch-size', [string]$saved.inference_batch_size)
Write-Output ((Get-Date).ToString('HH:mm') + ' Python 3.12 UTF-8; resume full agent; completed jobs reused; test 2014 unused')
if ($CheckOnly) {
    Write-Output ('Validated arguments: ' + ($experimentArgs -join ' '))
    exit 0
}
& rtk proxy py @experimentArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
