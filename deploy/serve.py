#!/usr/bin/env python3
"""Serve HADES over streamable HTTP with the ensemble already loaded.

`hades-mcp --http` loads the models on the first scoring call, which takes
20-60 seconds. Locally that is one slow call. As a hosted service it is worse:
the request that triggers it may time out, a health probe may declare the
container unready while it runs, and every replica pays it again after a scale
event. This loads them while the container starts, so the port opens only once
HADES can actually answer.

Everything else is `hades-mcp`: the same server, the same `hades_score` tool,
the same arguments and environment variables.
"""

from __future__ import annotations

import logging
import sys

import hades_mcp

log = logging.getLogger("hades-serve")


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    log.info("Warming the HADES ensemble before the port opens; this is slow.")
    # Populates hades_mcp's cached scorer -- the instance the tool itself will
    # use. Building a HADES_score here instead would warm a different object
    # and leave the tool to load its own on the first call. `_score_sync` is
    # private because nothing outside the module needed it until now; a `--warm`
    # flag in hades_mcp would be the tidier home for this.
    hades_mcp._score_sync(["CCO"])
    log.info("HADES is warm. Starting the HTTP server.")
    hades_mcp.main()


if __name__ == "__main__":
    main()
