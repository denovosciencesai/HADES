# Security policy

HADES loads its models from Python pickles, and unpickling a file runs code from
it. Problems in how those files are checked or loaded are security issues, and
should be reported privately rather than in a public issue.

## Reporting a vulnerability

Report privately through GitHub:
**Security → Report a vulnerability** on this repository
(<https://github.com/denovosciencesai/HADES/security/advisories/new>).

If you cannot use GitHub, email **<narekp@denovosciences.ai>**.

Please include what you found, the steps or files that reproduce it, and the
HADES version or commit you tested. We aim to acknowledge a report within
5 working days and to keep you informed until it is resolved. Please do not
disclose the issue publicly until a fix has been released.

## Supported versions

Only the latest release receives security fixes.

## What is in scope

- Getting a model file past the checksum check: a `.pkl` that `verify="require"`
  (or `--verify require`) accepts even though it does not match `SHA256SUMS`.
- Code execution from files that *do* pass verification against the shipped
  `SHA256SUMS`.
- Any other way that scoring or training runs code or writes files the user did
  not ask for.

## What is not in scope

- Loading a model directory that you obtained from a source you do not trust.
  As the README explains, a checksum only proves the bytes match the manifest.
  If the manifest came from the same untrusted source, it proves nothing, and
  the pickle runs on load either way.
- Running with `verify="off"`.
- Vulnerabilities in dependencies (RDKit, scikit-learn, PyTorch, ...). Report
  those upstream. Tell us too if HADES uses the affected code in a way that
  makes it exploitable.
