# Security Policy

AzLLM-Reliability is research software and is not intended to provide a
security boundary for production systems.

## Reporting a Vulnerability

Please report security-sensitive issues privately through GitHub's security
reporting features when available rather than opening a public issue.

Include:

- affected file or component,
- reproduction steps,
- expected impact,
- environment details,
- and any proposed mitigation.

## Sensitive Data

Do not commit:

- passwords,
- API keys,
- access tokens,
- private datasets,
- personally identifying information,
- model-provider credentials,
- or machine-specific secrets.

Model checkpoints and large generated artifacts should be distributed through
an appropriate artifact host rather than committed directly to Git history.
