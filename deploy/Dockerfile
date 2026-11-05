# HADES as a remote MCP server: streamable HTTP on /mcp, for Claude.ai custom
# connectors, for an organisation self-hosting it inside its own network, and
# for `docker run` on a laptop. The models ship inside the wheel, so the image
# needs nothing else at runtime and the container never downloads anything.
#
# Build without a local Docker daemon:
#   az acr build -r <registry> -t hades-mcp:0.9.1 -f deploy/Dockerfile .
#
# Run:
#   docker run --rm -p 8000:8000 hades-mcp:0.9.1
#   # then point an MCP client at http://localhost:8000/mcp

# The full image, not -slim. HADES never draws a molecule, but admet_ai imports
# chemfunc, whose __init__ imports smiles_to_svg, which imports rdMolDraw2D --
# so `from admet_ai import ADMETModel` drags in RDKit's renderer and its X11 and
# XML stack. On -slim that surfaces one missing .so at a time (libXrender, then
# libexpat, ...), each costing a build and a deploy to find. The base is ~800 MB
# against an image already over 5 GB of torch and scientific wheels, which is a
# cheap way to be rid of a whole class of failure.
FROM python:3.10

# Still explicit, because the list is the point: libgomp1 is OpenMP for LightGBM
# and XGBoost; the rest are what rdkit.Chem.Draw loads. Present in this base
# today, named here so a future move back to a smaller one cannot lose them
# silently.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgomp1 libxrender1 libxext6 libexpat1 libsm6 libice6 \
      libglib2.0-0 libfontconfig1 libfreetype6 \
 && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TQDM_DISABLE=1

# The release to serve. Keep it pinned: a hosted HADES that silently changes
# version changes the scores it reports, and nothing downstream would notice.
# CONTRIBUTING.md lists this with the other places the version lives.
ARG HADES_VERSION=0.9.1

# PyTorch's CPU-only build. PyPI's torch bundles several GB of CUDA libraries
# HADES never uses; environment.yml records that the CPU build scores the
# paper's test set bit-for-bit identically.
#
# setuptools<82 is no longer pinned here. It was, because the pin lived only in
# environment.yml and so never reached the published wheel's metadata, which a
# clean resolution like this one could then break. From 0.9.1 requirements.txt
# carries it, and pyproject.toml builds the wheel's dependencies from that
# file, so the constraint arrives with the package.
RUN pip install \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      "hades-druglikeness[mcp]==${HADES_VERSION}"

# Scoring writes nothing and needs no privileges.
RUN useradd --create-home --uid 10001 hades
USER hades
WORKDIR /home/hades

COPY --chown=hades:hades deploy/serve.py ./serve.py

# Score a molecule at build time, as the runtime user. The import chain that
# reaches RDKit's renderer runs here, so a missing system library fails the
# build in seconds instead of crash-looping a deployed revision -- which is how
# libXrender and libexpat were each found, one deploy at a time. It also proves
# the models are in the wheel and that the pinned stack imports at all.
RUN python -c "import hades_mcp; r = hades_mcp._score_sync(['CCO']); \
print('build smoke test:', r['results']); \
assert abs(r['results'][0]['hades_score'] - 0.017846526) < 1e-4, r"

# hades_mcp reads these when --http is given without --host/--port.
ENV HADES_HOST=0.0.0.0 \
    HADES_PORT=8000

EXPOSE 8000

# serve.py loads the ensemble before binding the port, so a container that is
# accepting connections is a container that can score. Expect 30-60 seconds
# between start and the first successful request.
CMD ["python", "serve.py", "--http"]
