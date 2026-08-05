# Security policy

## Supported versions

Duality is experimental research software. Security fixes are applied to the latest commit on `main`; older revisions are not maintained as supported releases.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting feature instead of opening a public issue. Include the affected file or component, reproduction steps, potential impact, and any suggested mitigation.

Do not include access tokens, model credentials, private datasets, or other secrets in reports, logs, examples, or test fixtures.

## Research-use considerations

- Review model licenses and dataset terms before running experiments.
- Treat downloaded model artifacts as untrusted input.
- Run large-model experiments in isolated environments with explicit resource limits.
- Pin dependencies for production or shared infrastructure deployments.
