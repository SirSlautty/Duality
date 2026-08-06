Set-StrictMode -Version Latest

function Assert-AdfsCommand {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Name)

    if (-not (Get-Command -Name $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' is not available. Run this on a server with the AD FS or Web Application Proxy role installed."
    }
}

function Get-CertificateRecord {
    [CmdletBinding()]
    param([Parameter(Mandatory)][System.Security.Cryptography.X509Certificates.X509Certificate2]$Certificate)

    [pscustomobject]@{
        Thumbprint = $Certificate.Thumbprint
        Subject = $Certificate.Subject
        Issuer = $Certificate.Issuer
        NotBefore = $Certificate.NotBefore.ToUniversalTime().ToString('o')
        NotAfter = $Certificate.NotAfter.ToUniversalTime().ToString('o')
        DaysRemaining = [math]::Floor(($Certificate.NotAfter - (Get-Date)).TotalDays)
        HasPrivateKey = $Certificate.HasPrivateKey
        EnhancedKeyUsage = @($Certificate.Extensions | Where-Object { $_.Oid.Value -eq '2.5.29.37' } | ForEach-Object { $_.Format($false) })
        DnsNames = @($Certificate.Extensions | Where-Object { $_.Oid.Value -eq '2.5.29.17' } | ForEach-Object { $_.Format($false) })
    }
}

function Get-LocalMachineCertificate {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Thumbprint)

    $normalized = ($Thumbprint -replace '\s', '').ToUpperInvariant()
    $certificate = Get-ChildItem -Path Cert:\LocalMachine\My | Where-Object { $_.Thumbprint -eq $normalized }
    if (-not $certificate) { throw "Certificate '$normalized' was not found in Cert:\LocalMachine\My." }
    return $certificate
}

function Test-AdfsCertificate {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][System.Security.Cryptography.X509Certificates.X509Certificate2]$Certificate,
        [string[]]$DnsName,
        [int]$MinimumDaysRemaining = 0,
        [switch]$RequirePrivateKey
    )

    $problems = [System.Collections.Generic.List[string]]::new()
    if ($Certificate.NotAfter -le (Get-Date).AddDays($MinimumDaysRemaining)) { $problems.Add("Certificate expires within $MinimumDaysRemaining days.") }
    if ($RequirePrivateKey -and -not $Certificate.HasPrivateKey) { $problems.Add('Certificate does not have an accessible private key.') }
    foreach ($name in @($DnsName)) {
        if (-not $Certificate.GetNameInfo([System.Security.Cryptography.X509Certificates.X509NameType]::DnsName, $false) -and $Certificate.Subject -notlike "*CN=$name*") {
            $problems.Add("Certificate subject/SAN does not appear to contain '$name'.")
        }
    }
    [pscustomobject]@{ IsValid = ($problems.Count -eq 0); Problems = @($problems) }
}

function Write-AdfsJson {
    [CmdletBinding()]
    param([Parameter(Mandatory)]$InputObject, [Parameter(Mandatory)][string]$Path)

    $parent = Split-Path -Parent $Path
    if ($parent -and -not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    $InputObject | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Path -Encoding UTF8
}

Export-ModuleMember -Function Assert-AdfsCommand,Get-CertificateRecord,Get-LocalMachineCertificate,Test-AdfsCertificate,Write-AdfsJson
