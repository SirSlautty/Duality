[CmdletBinding(SupportsShouldProcess, ConfirmImpact = 'High')]
param(
    [Parameter(Mandatory)][ValidatePattern('^[0-9A-Fa-f ]{40}$')][string]$Thumbprint,
    [switch]$UpdateServiceCommunications,
    [switch]$UpdateSsl,
    [string]$SnapshotPath = (Join-Path (Get-Location) ("adfs-before-{0:yyyyMMdd-HHmmss}.json" -f (Get-Date)))
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
Import-Module (Join-Path $PSScriptRoot 'AdfsAutomation.psm1')

if (-not ($UpdateServiceCommunications -or $UpdateSsl)) { throw 'Specify -UpdateServiceCommunications and/or -UpdateSsl. No certificate binding is changed by default.' }
Assert-AdfsCommand 'Get-AdfsCertificate'
$certificate = Get-LocalMachineCertificate -Thumbprint $Thumbprint
$validation = Test-AdfsCertificate -Certificate $certificate -MinimumDaysRemaining 1 -RequirePrivateKey
if (-not $validation.IsValid) { throw ($validation.Problems -join ' ') }

$snapshot = [pscustomobject]@{
    ComputerName = $env:COMPUTERNAME
    GeneratedUtc = (Get-Date).ToUniversalTime().ToString('o')
    AdfsCertificates = @(Get-AdfsCertificate | Select-Object CertificateType,Thumbprint,IsPrimary,@{n='NotAfter';e={$_.Certificate.NotAfter.ToUniversalTime().ToString('o')}})
    SslCertificates = if (Get-Command Get-AdfsSslCertificate -ErrorAction SilentlyContinue) { @(Get-AdfsSslCertificate | Select-Object Hostname,Port,Thumbprint) } else { @() }
}
Write-AdfsJson -InputObject $snapshot -Path $SnapshotPath

$normalized = ($Thumbprint -replace '\s', '').ToUpperInvariant()
if ($UpdateServiceCommunications -and $PSCmdlet.ShouldProcess($env:COMPUTERNAME, "bind $normalized as AD FS service communications certificate")) {
    Assert-AdfsCommand 'Set-AdfsCertificate'
    Set-AdfsCertificate -CertificateType Service-Communications -Thumbprint $normalized
}
if ($UpdateSsl -and $PSCmdlet.ShouldProcess($env:COMPUTERNAME, "bind $normalized as AD FS SSL certificate")) {
    Assert-AdfsCommand 'Set-AdfsSslCertificate'
    Set-AdfsSslCertificate -Thumbprint $normalized
}

Write-Output "Completed requested AD FS binding actions. Snapshot: $SnapshotPath"
