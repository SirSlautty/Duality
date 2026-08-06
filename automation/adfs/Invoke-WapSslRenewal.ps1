[CmdletBinding(SupportsShouldProcess, ConfirmImpact = 'High')]
param(
    [Parameter(Mandatory)][ValidatePattern('^[0-9A-Fa-f ]{40}$')][string]$Thumbprint,
    [string]$SnapshotPath = (Join-Path (Get-Location) ("wap-before-{0:yyyyMMdd-HHmmss}.json" -f (Get-Date)))
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
Import-Module (Join-Path $PSScriptRoot 'AdfsAutomation.psm1')

Assert-AdfsCommand 'Get-WebApplicationProxySslCertificate'
Assert-AdfsCommand 'Set-WebApplicationProxySslCertificate'
$certificate = Get-LocalMachineCertificate -Thumbprint $Thumbprint
$validation = Test-AdfsCertificate -Certificate $certificate -MinimumDaysRemaining 1 -RequirePrivateKey
if (-not $validation.IsValid) { throw ($validation.Problems -join ' ') }

$snapshot = [pscustomobject]@{
    ComputerName = $env:COMPUTERNAME
    GeneratedUtc = (Get-Date).ToUniversalTime().ToString('o')
    WapSslCertificate = @(Get-WebApplicationProxySslCertificate | Select-Object Hostname,Port,Thumbprint)
    WapConfiguration = if (Get-Command Get-WebApplicationProxyConfiguration -ErrorAction SilentlyContinue) { Get-WebApplicationProxyConfiguration | Select-Object * } else { $null }
}
Write-AdfsJson -InputObject $snapshot -Path $SnapshotPath

$normalized = ($Thumbprint -replace '\s', '').ToUpperInvariant()
if ($PSCmdlet.ShouldProcess($env:COMPUTERNAME, "bind $normalized as Web Application Proxy SSL certificate")) {
    Set-WebApplicationProxySslCertificate -Thumbprint $normalized
}
Write-Output "Completed WAP SSL binding action. Snapshot: $SnapshotPath"
