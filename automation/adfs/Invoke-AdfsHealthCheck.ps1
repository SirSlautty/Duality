[CmdletBinding()]
param(
    [string]$FederationServiceUrl,
    [int]$ExpiringWithinDays = 60,
    [int]$EventHours = 24,
    [string]$OutputPath
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
Import-Module (Join-Path $PSScriptRoot 'AdfsAutomation.psm1')

Assert-AdfsCommand 'Get-AdfsProperties'
Assert-AdfsCommand 'Get-AdfsCertificate'
$report = [pscustomobject]@{
    ComputerName = $env:COMPUTERNAME
    GeneratedUtc = (Get-Date).ToUniversalTime().ToString('o')
    Services = @(Get-Service adfssrv -ErrorAction SilentlyContinue | Select-Object Name,Status,StartType)
    Properties = Get-AdfsProperties | Select-Object AutoCertificateRollover,CertificateGenerationThreshold,CertificatePromotionThreshold,HostName
    Certificates = @(Get-AdfsCertificate | Select-Object CertificateType,Thumbprint,IsPrimary,@{n='NotAfter';e={$_.Certificate.NotAfter.ToUniversalTime().ToString('o')}},@{n='DaysRemaining';e={[math]::Floor(($_.Certificate.NotAfter - (Get-Date)).TotalDays)}})
    Endpoints = @(Get-AdfsEndpoint | Select-Object FullUrl,Protocol,Proxy,Enabled)
    RecentAdfsErrors = @()
    FederationService = $null
}

$since = (Get-Date).AddHours(-$EventHours)
$report.RecentAdfsErrors = @(Get-WinEvent -FilterHashtable @{ LogName = 'AD FS/Admin'; Level = 2; StartTime = $since } -ErrorAction SilentlyContinue | Select-Object -First 50 TimeCreated,Id,ProviderName,Message)
if ($FederationServiceUrl) {
    try {
        $response = Invoke-WebRequest -Uri $FederationServiceUrl -UseBasicParsing -Method Head -TimeoutSec 20
        $report.FederationService = [pscustomobject]@{ Url = $FederationServiceUrl; StatusCode = [int]$response.StatusCode; Healthy = $true }
    } catch {
        $report.FederationService = [pscustomobject]@{ Url = $FederationServiceUrl; Healthy = $false; Error = $_.Exception.Message }
    }
}

$report | Add-Member -NotePropertyName ExpiringCertificates -NotePropertyValue @($report.Certificates | Where-Object { $_.DaysRemaining -le $ExpiringWithinDays })
if ($OutputPath) { Write-AdfsJson -InputObject $report -Path $OutputPath }
$report
