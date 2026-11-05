#!/usr/bin/env python3
"""Serve HADES over streamable HTTP, warm and stateless.

Two differences from `hades-mcp --http`. Both matter only when the server is
hosted rather than started beside the client, which is why they live here and
not in `hades_mcp.py`.

**Warm.** The models load on the first scoring call: 20-60 seconds. Hosted,
that is the request that times out, the health probe that fires mid-load, and
a cost every replica pays again after a scale event. This loads them before
binding the port, so a container that accepts connections is a container that
can score.

**Stateless.** The SDK keeps sessions in the process's memory and answers an
unknown `Mcp-Session-Id` with 404. Behind a load balancer with more than one
replica and no session affinity, a client's second request can reach a replica
that never saw its first, and the call fails -- intermittently, only under
load, and looking like an unreliable server rather than a routing problem.
Every deployment orphans open sessions the same way. `hades_score` is a pure
function of its arguments: there is no session state worth keeping, so nothing
is lost by not keeping it, and the server stops depending on which replica a
request reaches.

Everything else is `hades-mcp`: the same server, the same tool, the same
`HADES_*` environment variables.
"""

from __future__ import annotations

import logging
import os
import sys

import anyio

import hades_mcp

log = logging.getLogger("hades-serve")

HOST = os.environ.get("HADES_HOST") or "0.0.0.0"  # noqa: S104 -- a container port
PORT = int(os.environ.get("HADES_PORT") or 8000)


def main() -> None:
    os.environ.setdefault("TQDM_DISABLE", "1")
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    log.info("Warming the HADES ensemble before the port opens; this is slow.")
    # Populates hades_mcp's cached scorer -- the instance the tool itself will
    # use. Building a HADES_score here instead would warm a different object
    # and leave the tool to load its own on the first call. `_score_sync` is
    # private because nothing outside the module needed it until now; a
    # `--warm` flag in hades_mcp would be the tidier home for this.
    hades_mcp._score_sync(["CCO"])
    log.info("HADES is warm. Serving on %s:%d/mcp, stateless.", HOST, PORT)

    anyio.run(
        lambda: hades_mcp.mcp.run_streamable_http_async(
            host=HOST, port=PORT, stateless_http=True
        )
    )


if __name__ == "__main__":
    main()
