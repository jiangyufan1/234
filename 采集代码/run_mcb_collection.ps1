# Forward all options to the position/orientation/contact collector.
# Example: .\run_mcb_collection.ps1 --run --headless --max-configs 2
$collectionPython = Join-Path $env:USERPROFILE '.conda\envs\mujoco\python.exe'
if ($env:TACTILE_PYTHON) {
    $collectionPython = $env:TACTILE_PYTHON
}
if (-not (Test-Path -LiteralPath $collectionPython -PathType Leaf)) {
    $collectionPython = (Get-Command python -ErrorAction Stop).Source
}
$previousIOEncoding = $env:PYTHONIOENCODING
try {
    $env:PYTHONIOENCODING = 'utf-8'
    & $collectionPython (Join-Path $PSScriptRoot 'collect_mcb_position_pose.py') @args
    $collectionExitCode = $LASTEXITCODE
}
finally {
    $env:PYTHONIOENCODING = $previousIOEncoding
}
exit $collectionExitCode
