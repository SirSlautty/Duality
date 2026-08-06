[CmdletBinding()]
param([int]$WarningDays = 60, [string]$OutputPath)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
Import-Module (Join-Path $PSScriptRoot 'AdfsAutomation.psm1')
Assert-AdfsCommand 'Get-AdfsCertificate'
Assert-AdfsCommand 'Get-AdfsProperties'

$result = [pscustomobject]@{
    ComputerName = $env:COMPUTERNAME
    GeneratedUtc = (Get-Date).ToUniversalTime().ToString('o')
    AutoCertificateRollover = (Get-AdfsProperties).AutoCertificateRollover
    TokenSigning = @(Get-AdfsCertificate -CertificateType Token-Signing | Select-Object Thumbprint,IsPrimary,@{n='NotAfter';e={$_.Certificate.NotAfter.ToUniversalTime().ToString('o')}},@{n='DaysRemaining';e={[math]::Floor(($_.Certificate.NotAfter - (Get-Date)).TotalDays)}})
    TokenDecrypting = @(Get-AdfsCertificate -CertificateType Token-Decrypting | Select-Object Thumbprint,IsPrimary,@{n='NotAfter';e={$_.Certificate.NotAfter.ToUniversalTime().ToString('o')}},@{n='DaysRemaining';e={[math]::Floor(($_.Certificate.NotAfter - (Get-Date)).TotalDays)}})
}
$result | Add-Member -NotePropertyName Expiring -NotePropertyValue @($result.TokenSigning + $result.TokenDecrypting | Where-Object { $_.DaysRemaining -le $WarningDays })
if ($OutputPath) { Write-AdfsJson -InputObject $result -Path $OutputPath }
$result
