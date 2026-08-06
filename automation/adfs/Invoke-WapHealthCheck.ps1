[CmdletBinding()]
param(
    [string]$FederationServiceUrl,
    [int]$EventHours = 24,
    [string]$OutputPath
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
Import-Module (Join-Path $PSScriptRoot 'AdfsAutomation.psm1')

Assert-AdfsCommand 'Get-WebApplicationProxyConfiguration'
Assert-AdfsCommand 'Get-WebApplicationProxySslCertificate'
$report = [pscustomobject]@{
    ComputerName = $env:COMPUTERNAME
    GeneratedUtc = (Get-Date).ToUniversalTime().ToString('o')
    Services = @(Get-Service WAPCS -ErrorAction SilentlyContinue | Select-Object Name,Status,StartType)
    Configuration = Get-WebApplicationProxyConfiguration | Select-Object *
    SslCertificate = @(Get-WebApplicationProxySslCertificate | Select-Object Hostname,Port,Thumbprint)
    Applications = if (Get-Command Get-WebApplicationProxyApplication -ErrorAction SilentlyContinue) { @(Get-WebApplicationProxyApplication | Select-Object Name,ExternalUrl,BackendServerUrl,ExternalCertificateThumbprint) } else { @() }
    RecentErrors = @()
    FederationService = $null
}

$since = (Get-Date).AddHours(-$EventHours)
$wapLogNames = @(Get-WinEvent -ListLog '*WebApplicationProxy*' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty LogName)
$report.RecentErrors = @($wapLogNames | ForEach-Object { Get-WinEvent -FilterHashtable @{ LogName = $_; Level = 2; StartTime = $since } -ErrorAction SilentlyContinue } | Select-Object -First 50 TimeCreated,Id,ProviderName,Message)
if ($FederationServiceUrl) {
    try {
        $response = Invoke-WebRequest -Uri $FederationServiceUrl -UseBasicParsing -Method Head -TimeoutSec 20
        $report.FederationService = [pscustomobject]@{ Url = $FederationServiceUrl; StatusCode = [int]$response.StatusCode; Healthy = $true }
    } catch {
        $report.FederationService = [pscustomobject]@{ Url = $FederationServiceUrl; Healthy = $false; Error = $_.Exception.Message }
    }
}
if ($OutputPath) { Write-AdfsJson -InputObject $report -Path $OutputPath }
$report
