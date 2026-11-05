# Privacy policy for the hosted HADES service

**Last updated: 5 November 2026**

This policy covers the hosted HADES service at `https://hades.denovosciences.ai/mcp`,
operated by Denovo Sciences.

## Scope

This policy covers the hosted service only.

HADES also runs on your own machine. The Claude Code plugin, the `hades-mcp`
command and the `hades-score` command all score molecules locally and do not
transmit your structures, and the same container that runs the hosted service
can be run inside your own network (see
[deploy/README.md](https://github.com/denovosciencesai/HADES/blob/main/deploy/README.md)).
Those uses send nothing to us, so nothing in this policy applies to them.

If you reach the hosted service through Claude, your conversation — including
any structure you paste into it — is handled by Anthropic under Anthropic's own
terms before it reaches us. This policy covers only what happens from the moment
a request arrives at our endpoint.

## What we collect

The hosted service receives one thing from you: the SMILES strings you ask it to
score. It asks for no personal data.

Those SMILES are held in memory for the duration of the request, used to compute
a score, and returned to you. They are not written to disk, not recorded in any
log we keep, and not retained after the response is sent.

One current exception, stated because it is true today: if a string cannot be
parsed as SMILES at all, the chemistry library echoes it, together with the parse
error, to the container's console output, and the hosting platform keeps that
output readable to us for a period even though no log destination is configured.
This affects only input the parser rejects. We do not collect anything from that
output.

**We do not use submitted structures to train, tune or improve the model, or for
any purpose other than returning your score.** The model is fixed and versioned;
see [model-integrity.md](model-integrity.md).

The service has no accounts, no sign-in and no API keys. We do not ask for your
name, email address or organisation, and the Model Context Protocol, as this
service uses it, carries no user identity — the only client information sent is
the name and version of the MCP client software.

We also do not:

- store IP addresses (the one qualification is under Analytics below);
- use cookies, tracking pixels or advertising identifiers;
- build profiles of users.

## Who processes your data, and where

Denovo Sciences is responsible for the hosted service (the data controller,
where that term applies). Two infrastructure providers process traffic on our
behalf, under their standard data processing terms:

| Provider | Role | Location |
| --- | --- | --- |
| Microsoft Azure | Runs the scoring container | North Europe (Ireland) |
| Cloudflare | Terminates TLS at the network edge, applies rate limiting, produces aggregate traffic statistics | Global edge network |

**Cloudflare decrypts requests at its edge.** It means your SMILES exist in
memory, unencrypted, on Cloudflare infrastructure for the moment it takes to
forward them to the scoring service. We state this plainly because it is the
fact most likely to matter to you. Running HADES locally or hosting it yourself
avoids this; both run the same model and return the same scores.

Traffic between you and Cloudflare, and between Cloudflare and the scoring
service, is encrypted in transit with TLS. Because Cloudflare serves from the
edge location nearest the client, connection metadata may be processed outside
the UK and EEA; this is covered by Cloudflare's data processing terms.

Their processing on our behalf is governed by the
[Cloudflare Data Processing Addendum](https://www.cloudflare.com/cloudflare-customer-dpa/)
and the
[Microsoft Products and Services Data Protection Addendum](https://www.microsoft.com/licensing/docs/view/Microsoft-Products-and-Services-Data-Protection-Addendum-DPA).

## Who we share data with

We share your requests with no one other than the two infrastructure providers
named above, who process them on our behalf under the terms linked above.

We do not sell data. We do not share it with advertisers, data brokers,
analytics companies, research partners or any other third party, for any
purpose. We would disclose data to anyone else only if the law required it, and
since we retain none, there would be little to disclose.

## Analytics

We look at aggregate traffic statistics produced by Cloudflare: how many
requests arrive, when, and which countries they come from. Cloudflare derives
the country from the connecting IP address at its edge; we receive aggregate
counts.

Two honest qualifications:

- Cloudflare's security dashboard can show us individual requests that its rate
  limiting blocked, including the IP address they came from, within its
  retention window. We do not export, keep or act on these beyond operating the
  service.
- When you reach the service through Claude rather than directly, requests
  arrive from Anthropic's infrastructure, so the country we see is Anthropic's,
  not yours.

The Cloudflare plan the service runs on does not offer per-request logging
(Logpush), so the absence of request logs at the edge is a property of the
configuration, not only a policy we have chosen. Our own container has no log
destination configured.

## Retention

Request content is not retained: it exists only in memory while the request is
served, with the one exception of the parse-error echo described under
"What we collect".

Cloudflare retains its analytics and security event data for at least 30 days,
on a schedule Cloudflare sets. Those datasets are built from connection
metadata, not request bodies, so they are not where SMILES or scores would
appear.

## Security

The scoring service accepts connections only from Cloudflare's published network
ranges, so it cannot be reached directly from the public internet. The scoring
process runs as an unprivileged user, writes nothing to disk, and holds no
credentials. Rate limiting is applied per client address, at 10 requests per 10
seconds.

## Your rights

Denovo Sciences is established in the United States. If the UK GDPR or the EU
GDPR applies to your use of this service, you have rights of access,
rectification, erasure, restriction, objection and portability over personal
data held about you.

Because the hosted service retains almost nothing about you — at most, an IP
address in Cloudflare's security events if your requests were rate-limited — in
normal operation there is little held about you to which these rights could
attach.

Where that law applies, the transient processing of connection metadata for
rate limiting and security rests on our legitimate interest in operating the
service reliably (Article 6(1)(f)).

Scoring itself runs in Ireland, so that is where the structures you send are
scored. Cloudflare serves from the edge nearest you, so connection metadata, and
the request itself for the moment described above, may pass through Cloudflare
infrastructure elsewhere, including outside the UK and EEA, under Cloudflare's
data processing terms.

If you believe we hold personal data about you, write to the address below and
we aim to respond within one month. You may also complain to your national data
protection authority; in the UK that is the
[Information Commissioner's Office](https://ico.org.uk/).

## Confidentiality is not privacy

Chemical structures are not usually personal data, so data protection law may
have little to say about them. They may nonetheless be commercially sensitive or
unpublished. Nothing in this policy is a confidentiality undertaking for the
hosted service. If your structures are subject to a confidentiality obligation,
run HADES locally or inside your own network, where they are never sent to us.

## Changes

Material changes to this policy will be published here, and the date at the top
updated. The published history of this file is in the repository's git history.

## Contact

Denovo Sciences, Inc., a Delaware corporation (United States)
— info@denovosciences.ai

For questions about HADES itself, including bug reports, use
[GitHub issues](https://github.com/denovosciencesai/HADES/issues). Please do not
put confidential structures in an issue.
