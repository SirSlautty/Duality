[CmdletBinding()]
param(
    [int]$ExpiringWithinDays = 60,
    [string]$OutputPath
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
Import-Module (Join-Path $PSScriptRoot 'AdfsAutomation.psm1')

Assert-AdfsCommand 'Get-AdfsCertificate'
Assert-AdfsCommand 'Get-AdfsSslCertificate'

$records = [System.Collections.Generic.List[object]]::new()
foreach ($type in 'Service-Communications','Token-Signing','Token-Decrypting') {
    foreach ($entry in @(Get-AdfsCertificate -CertificateType $type)) {
        $records.Add([pscustomobject]@{
            Scope = 'ADFS'
            CertificateType = $type
            IsPrimary = $entry.IsPrimary
            Thumbprint = $entry.Thumbprint
            NotAfter = $entry.Certificate.NotAfter.ToUniversalTime().ToString('o')
            DaysRemaining = [math]::Floor(($entry.Certificate.NotAfter - (Get-Date)).TotalDays)
            HasPrivateKey = $entry.Certificate.HasPrivateKey
            Subject = $entry.Certificate.Subject
        })
    }
}

foreach ($entry in @(Get-AdfsSslCertificate)) {
    $records.Add([pscustomobject]@{
        Scope = 'ADFS'
        CertificateType = 'SSL'
        IsPrimary = $null
        Hostname = $entry.Hostname
        Port = $entry.Port
        Thumbprint = $entry.Thumbprint
        NotAfter = $entry.Certificate.NotAfter.ToUniversalTime().ToString('o')
        DaysRemaining = [math]::Floor(($entry.Certificate.NotAfter - (Get-Date)).TotalDays)
        HasPrivateKey = $entry.Certificate.HasPrivateKey
        Subject = $entry.Certificate.Subject
    })
}

$report = [pscustomobject]@{
    ComputerName = $env:COMPUTERNAME
    GeneratedUtc = (Get-Date).ToUniversalTime().ToString('o')
    ExpiringWithinDays = $ExpiringWithinDays
    Certificates = @($records)
    Expiring = @($records | Where-Object { $_.DaysRemaining -le $ExpiringWithinDays })
}

if ($OutputPath) { Write-AdfsJson -InputObject $report -Path $OutputPath }
$report
