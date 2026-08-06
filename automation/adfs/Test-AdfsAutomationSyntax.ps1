[CmdletBinding()]
param([string]$Path = $PSScriptRoot)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$errors = [System.Collections.Generic.List[string]]::new()
foreach ($file in Get-ChildItem -LiteralPath $Path -Filter '*.ps1' -File) {
    $tokens = $null
    $parseErrors = $null
    [System.Management.Automation.Language.Parser]::ParseFile($file.FullName, [ref]$tokens, [ref]$parseErrors) | Out-Null
    foreach ($error in @($parseErrors)) { $errors.Add("$($file.Name): $($error.Message)") }
}
if ($errors.Count) { $errors | ForEach-Object { Write-Error $_ }; exit 1 }
Write-Output "PowerShell syntax OK: $((Get-ChildItem -LiteralPath $Path -Filter '*.ps1' -File).Count) scripts"
