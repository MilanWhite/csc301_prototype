$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& "$PSScriptRoot/.venv/Scripts/python.exe" "$PSScriptRoot/web_app.py" @args
