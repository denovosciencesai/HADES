# HADES as a remote MCP server

`hades_mcp.py` already serves streamable HTTP (`hades-mcp --http`). This
directory packages that into a container, so HADES can run as a remote MCP
server rather than a local one:

- a **Claude.ai custom connector**, for the web and mobile apps, which cannot
  start a local process;
- **self-hosted inside an organisation**, where structures must not leave the
  network — the same image, on the customer's own infrastructure;
- `docker run`, for anyone who wants HADES without installing it.

Local MCP clients do not need any of this. Claude Code, Claude Desktop and
other clients that launch a local process should use the plugin or `uvx`; see
[docs/mcp.md](../docs/mcp.md). Scoring locally keeps every structure on the
machine, which for unpublished chemistry is usually the deciding argument.

## What is in here

| File | |
| --- | --- |
| `Dockerfile` | The image: HADES from PyPI, CPU-only torch, models inside the wheel |
| `serve.py` | Loads the ensemble, then starts the HTTP server |

## Build

No local Docker daemon needed — Azure Container Registry builds it:

```bash
az acr build -r <registry> -g <resource-group> \
    -t hades-mcp:0.9.0 -f deploy/Dockerfile --platform linux/amd64 .
```

or, with Docker:

```bash
docker build -t hades-mcp:0.9.0 -f deploy/Dockerfile .
docker run --rm -p 8000:8000 hades-mcp:0.9.0
# the MCP endpoint is then http://localhost:8000/mcp
```

`HADES_VERSION` is a build argument, pinned to a release rather than tracking
the newest. A hosted scorer that silently changes version changes the numbers
it reports, and nothing downstream would notice.

The build **scores ethanol as its last step** and fails if the result is not
0.017846526. That is not ceremony: the import chain
`admet_ai -> chemfunc -> smiles_to_svg -> rdMolDraw2D` pulls RDKit's renderer
and its X11 and XML libraries into a purely numerical pipeline, and on a slim
base each missing library surfaced only as a crash-looping deployment, one
build-and-deploy cycle at a time. Failing the build instead takes seconds. Keep
this step.

## Why `serve.py` rather than `hades-mcp --http`

The models load on the first scoring call: 20-60 seconds. Hosted, that is the
request that times out, the health probe that fires mid-load, and a cost paid
again by every replica after a scale event. `serve.py` loads them before
binding the port, so a container that accepts connections is a container that
can score. Expect 30-60 seconds between container start and the first
successful request.

## Deploying to Azure Container Apps

```bash
az containerapp create -n hades-mcp -g <resource-group> --environment <env> \
    --image <registry>.azurecr.io/hades-mcp:0.9.0 \
    --registry-server <registry>.azurecr.io --registry-identity system \
    --target-port 8000 --ingress external --transport http \
    --cpu 2 --memory 4Gi --min-replicas 1 --max-replicas 2
```

`--min-replicas 1` is deliberate. Scaling to zero is nearly free, but every
cold start pays the model load again, which is what `serve.py` exists to avoid.

Verified on 2 vCPU / 4 GiB: the container starts, warms and serves with no
restarts. A scoring call for four molecules round-trips in about six seconds
from outside Azure, and the scores match a local HADES exactly.

## Security

**The server has no authentication of its own.** `run_streamable_http_async`
serves whoever connects, so anything in front of it is the only control there
is. Until that is in place, restrict it by address:

```bash
az containerapp ingress access-restriction set -n hades-mcp -g <resource-group> \
    --rule-name only-operator --ip-address <your-ip>/32 --action Allow
```

An open endpoint is a scoring service anyone can bill to your subscription.
Before removing the restriction, put a token or OAuth in front of it; a listing
in Anthropic's connector directory requires OAuth in any case.

Scoring is read-only and writes nothing, the container runs as a non-root user,
and no structure is logged — but a hosted HADES does mean SMILES leave the
submitter's machine. Say so plainly to anyone you offer it to, and expect
pharmaceutical users to prefer self-hosting for exactly that reason.
