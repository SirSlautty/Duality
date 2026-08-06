# AD FS and Web Application Proxy automation

This directory contains small, auditable PowerShell scripts for certificate inventory, preflight checks, renewal, and post-change health checks in AD FS and Web Application Proxy (WAP) environments.

The scripts are intentionally conservative:

- Inventory, token-certificate status, health checks, and syntax checks are read-only.
- Renewal scripts require the target thumbprint and an explicit action switch.
- Renewal scripts support `-WhatIf` and write a JSON snapshot before changing a binding.
- Token-signing and token-decrypting rollover is not hidden inside the SSL renewal path. Those certificates are consumed by federation partners and must be handled as a planned change.
- No script exports private keys or AD FS secrets.

## Recommended runbook

1. On each AD FS and WAP server, import the replacement certificate into `Cert:\LocalMachine\My`. Confirm the private key exists and the AD FS service account can read it.
2. Run the inventory and health checks, saving their JSON output as change evidence:

   ```powershell
   .\Get-AdfsCertificateInventory.ps1 -OutputPath .\reports\adfs-before.json
   .\Get-AdfsTokenCertificateStatus.ps1 -OutputPath .\reports\tokens.json
   .\Invoke-AdfsHealthCheck.ps1 -FederationServiceUrl https://fs.example.com/adfs/ls/IdpInitiatedSignOn.aspx -OutputPath .\reports\adfs-health.json
   .\Invoke-WapHealthCheck.ps1 -FederationServiceUrl https://fs.example.com/adfs/ls/IdpInitiatedSignOn.aspx -OutputPath .\reports\wap-health.json
   ```

3. Preview the SSL/service-communications change:

   ```powershell
   .\Invoke-AdfsSslRenewal.ps1 -Thumbprint '0123456789ABCDEF0123456789ABCDEF01234567' -UpdateSsl -UpdateServiceCommunications -WhatIf
   ```

4. Apply the AD FS change during a maintenance window, then run the health check again.
5. Apply the WAP SSL binding on one WAP server at a time and validate external federation before continuing:

   ```powershell
   .\Invoke-WapSslRenewal.ps1 -Thumbprint '0123456789ABCDEF0123456789ABCDEF01234567' -WhatIf
   .\Invoke-WapSslRenewal.ps1 -Thumbprint '0123456789ABCDEF0123456789ABCDEF01234567'
   ```

6. For token signing/decryption certificates, coordinate relying-party and claims-provider updates first. Use the status script to monitor primary and secondary certificates; do not use an SSL renewal script for token rollover.

## Script reference

| Script | Purpose | Writes configuration? |
| --- | --- | --- |
| `Get-AdfsCertificateInventory.ps1` | AD FS service, token, and SSL certificate inventory | No |
| `Get-AdfsTokenCertificateStatus.ps1` | Token certificate expiry and rollover posture | No |
| `Invoke-AdfsHealthCheck.ps1` | Service, AD FS properties, endpoints, recent admin errors, optional URL probe | No |
| `Invoke-WapHealthCheck.ps1` | WAP service, configuration, SSL binding, published applications, recent errors, optional URL probe | No |
| `Invoke-AdfsSslRenewal.ps1` | Bind an imported certificate to AD FS SSL and/or service communications | Yes, explicit switches only |
| `Invoke-WapSslRenewal.ps1` | Bind an imported certificate to WAP SSL | Yes, explicit invocation only |
| `Test-AdfsAutomationSyntax.ps1` | Parse all scripts without requiring AD FS role cmdlets | No |

Run the scripts elevated on the appropriate Windows Server. AD FS and WAP cmdlets are discovered at runtime so syntax validation can run on a non-AD-FS workstation.

The underlying cmdlet and rollover behavior follows Microsoft’s [AD FS certificate guidance](https://learn.microsoft.com/en-us/windows-server/identity/ad-fs/operations/configure-ts-td-certs-ad-fs), [AD FS SSL troubleshooting guidance](https://learn.microsoft.com/en-us/troubleshoot/windows-server/active-directory/troubleshoot-ad-fs-sso-issue), and [Web Application Proxy documentation](https://learn.microsoft.com/en-us/windows-server/remote/remote-access/web-application-proxy/web-app-proxy-windows-server).

## Safety notes

The snapshot files contain certificate metadata and AD FS configuration values, not private keys. Treat them as operational data. Review the generated snapshot before applying a change, keep one copy outside the server, and retain the old certificate until post-change validation is complete.

Token-signing and token-decrypting changes can invalidate relying parties or encrypted assertions. Follow the Microsoft rollover process and allow partners to consume the secondary certificate before promotion. The scripts here provide visibility and evidence; they deliberately do not automate partner coordination or emergency rotation.
