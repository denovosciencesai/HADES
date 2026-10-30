#!/usr/bin/env python3
"""MCP server for HADES: SMILES in, HADES oral drug-likeness score out.

Exposes HADES as a `hades_score` tool to any Model Context Protocol client --
Claude Code (where this repository installs as a plugin), Claude Desktop,
Cursor, VS Code and others. Serve it over stdio, the transport those clients
launch it with:

    python hades_mcp.py

Check it without an MCP client (scores a few molecules, reports on stderr):

    python hades_mcp.py --selftest
    python hades_mcp.py --selftest "CCO" "c1ccccc1"

Configuration, all optional:
    HADES_REPO        The HADES checkout to import HADES and its shipped models
                      from. Defaults to the directory of this file, so nothing
                      needs setting while it stays in the repo.
    HADES_MODELS_DIR  Directory of model .pkl files, to score with an ensemble
                      you trained yourself. Defaults to the shipped models.
    HADES_X_TRAIN     The X_train.parquet that goes with HADES_MODELS_DIR.
                      Defaults to the one that goes with the shipped models.
    HADES_VERIFY      Checksum check before unpickling the models: auto
                      (default), require or off -- the same choice as
                      `HADES_CALCULATOR.py --verify`.
    HADES_MAX_BATCH   Maximum SMILES per call (default 100, which keeps a full
                      result well inside the output limits MCP clients apply).

The ensemble and the featurizer load once, on the first scoring call, and are
reused for the life of the process: loading is slow (chemprop checkpoints, five
pickled models), scoring afterwards is not.

On stdio, stdout is the JSON-RPC channel, and the scientific stack under HADES
(chemprop, admet_ai, LightGBM, mordred) prints freely, some of it from compiled
code. The MCP SDK takes care of that: while it serves stdio it points fd 1 at
stderr and talks to the client through a private duplicate, so stray writes land
on stderr. `main()` adds one thing, line-buffering `sys.stdout`, so a print
reaches fd 1 while it still points at stderr; block-buffered, it would sit in
the buffer until exit and be flushed after the SDK has handed fd 1 back to the
client. Do not replace `sys.stdout` before `mcp.run()` -- the SDK then serves
it in place, and any print reaching it corrupts the stream.

Needs the `hades` conda environment (environment.yml) plus the MCP SDK:
pip install -e ".[mcp]"
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Annotated, Any

try:
    import anyio
    from mcp.server import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import ToolAnnotations
    from pydantic import BaseModel, Field
except ImportError as exc:
    raise ImportError(
        f"{exc}. The HADES MCP server needs the MCP Python SDK; install it into "
        'the hades environment with: pip install -e ".[mcp]"') from exc

log = logging.getLogger("hades-mcp")

REPO_ROOT = Path(os.environ.get("HADES_REPO") or Path(__file__).resolve().parent)
VERIFY = os.environ.get("HADES_VERIFY", "auto")
MAX_BATCH = int(os.environ.get("HADES_MAX_BATCH") or 100)  # empty means unset, like the others

mcp = MCPServer(
    name="hades",
    instructions=(
        "Scores small molecules with HADES (Holistic AI-based Drug-likeness "
        "Estimation Score), a soft-voting ensemble of five tree and gradient-"
        "boosting classifiers over Mordred, ADMET-AI and QED descriptors that "
        "predicts oral drug-likeness from a SMILES string. Use the hades_score "
        "tool whenever a molecule needs a HADES score."
    ),
)

_model_lock = threading.Lock()
_hades: Any = None


class Scored(BaseModel):
    index: int = Field(description="Position of the molecule in the input list")
    smiles: str = Field(description="The SMILES as supplied, whitespace stripped")
    hades_score: float = Field(description="HADES score in [0, 1]")


class Unscored(BaseModel):
    index: int = Field(description="Position of the input in the input list")
    smiles: str = Field(description="The input as supplied")
    reason: str = Field(description="Why it could not be scored")


class ScoreReport(BaseModel):
    results: list[Scored] = Field(description="One entry per scored molecule, in input order")
    invalid: list[Unscored] = Field(description="One entry per input that could not be scored")


def _importable():
    """Put the repository on sys.path, so HADES_CALCULATOR and HADES import from it."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _model_paths() -> dict[str, str]:
    """The model and descriptor paths to pass to HADES_score.

    The server runs in the client's working directory -- the user's project, not
    this repository -- and HADES_score's default paths are relative to the
    working directory. So pass the repository's own models and the training
    parquet that goes with them, as absolute paths, then let HADES_MODELS_DIR
    and HADES_X_TRAIN override either one.

    A checkout whose HADES_saved_models/ holds no models -- the package ships
    them itself, or only a stray file is left there -- gets HADES_score's
    defaults instead, which then resolve inside the package.
    """
    paths = {}
    saved_models = REPO_ROOT / "HADES_saved_models"
    if any(saved_models.glob("*.pkl")):
        paths["models_dir"] = str(saved_models)
        paths["x_train_parquet"] = str(REPO_ROOT / "train_test_saved" / "X_train.parquet")
    if os.environ.get("HADES_MODELS_DIR"):
        paths["models_dir"] = os.environ["HADES_MODELS_DIR"]
    if os.environ.get("HADES_X_TRAIN"):
        paths["x_train_parquet"] = os.environ["HADES_X_TRAIN"]
    return paths


def _load_hades() -> Any:
    """Build the HADES scorer. Caller must hold _model_lock."""
    paths = _model_paths()
    log.info("Loading HADES (first call only, this is slow): %s", paths)
    _importable()
    from HADES_CALCULATOR import HADES_score  # noqa: PLC0415 -- deliberately lazy

    try:
        scorer = HADES_score(quiet=True, verify=VERIFY, **paths)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"{exc}. Set HADES_REPO to the HADES repository root, or HADES_MODELS_DIR "
            "and HADES_X_TRAIN to a model you trained.") from exc
    log.info("HADES ready: %d ensemble members.", len(scorer.models))
    return scorer


def _score_sync(requested: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Validate, featurize and score. Runs in a worker thread; serialized by _model_lock.

    Inputs are checked with the same `unscoreable_reason` gate the featurizer
    applies, so everything that passes it gets a score. `score()` returns the
    scored SMILES without their input positions; checking that they come back
    exactly as sent is what keeps each score attached to its molecule.

    The gate looks at the molecule as written, but HADES scores it after
    stripping salts. An input that is nothing but salts or counter-ions (NaCl,
    acetic acid, water) is stripped to an empty molecule, every descriptor
    comes back missing, and mean-imputation turns that into the same score
    for all of them -- so such inputs are rejected here as well.
    """
    global _hades
    with _model_lock:
        _importable()
        from rdkit import Chem  # noqa: PLC0415 -- deliberately lazy
        from HADES.featurize import Featurizer, unscoreable_reason  # noqa: PLC0415

        valid: list[tuple[int, str]] = []
        invalid: list[dict[str, Any]] = []
        for index, raw in enumerate(requested):
            candidate = raw.strip()
            reason = unscoreable_reason(candidate)
            if reason is None and Featurizer.standartize_mol(
                    Chem.MolFromSmiles(candidate)).GetNumAtoms() == 0:
                reason = "nothing is left once salts and counter-ions are stripped"
            if reason is None:
                valid.append((index, candidate))
            else:
                invalid.append({"index": index, "smiles": raw, "reason": reason})

        if not valid:
            return {"results": [], "invalid": invalid}
        if _hades is None:
            _hades = _load_hades()
        frame = _hades.score([smiles for _, smiles in valid])

    if frame["SMILES"].tolist() != [smiles for _, smiles in valid]:
        raise RuntimeError(f"HADES scored {len(frame)} of {len(valid)} valid molecules; "
                           "refusing to report possibly misaligned results.")
    return {
        "results": [{"index": index, "smiles": smiles, "hades_score": float(score)}
                    for (index, smiles), score in zip(valid, frame["HADES_Score"])],
        "invalid": invalid,
    }


@mcp.tool(
    name="hades_score",
    title="HADES drug-likeness score",
    description=(
        "Compute the HADES oral drug-likeness score for one molecule or a batch, "
        "given SMILES. Call this whenever a HADES score, HADES prediction, or "
        "HADES ranking is requested for a structure. The score is the mean "
        "predicted positive-class probability of a five-model ensemble, in "
        "[0, 1]; higher means more similar to the approved oral drugs HADES "
        "was trained on. Takes a list of SMILES strings (a one-item list for a "
        "single molecule); pass all the molecules to score in one call, up to "
        f"{MAX_BATCH}. Inputs that cannot be scored (unparseable, empty, "
        "fragments with attachment points, or nothing but salts) are reported "
        "in 'invalid' with a reason rather than failing the whole call. Salt "
        "forms are scored as their parent molecule. Each entry carries "
        "'index', its position in the input. A low "
        "score for chemistry far outside the training distribution is not "
        "evidence that a molecule is not drug-like."
    ),
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
)
async def hades_score(
    smiles: Annotated[list[str], Field(description="SMILES strings to score")],
) -> ScoreReport:
    """Score molecules with HADES.

    The parameter is a plain list rather than "a string or a list": some
    clients drop `anyOf` schemas and would show the model an untyped argument.
    Errors are raised as `ToolError`, the one exception type whose message the
    SDK passes on to the model; any other exception reaches it as a bare
    "Error executing tool".

    Args:
        smiles: SMILES strings to score.

    Returns:
        ScoreReport: `results` holds one entry per scored molecule, in input
            order; `invalid` one entry per input that could not be scored,
            empty when every input scored.
    """
    if not smiles:
        raise ToolError("No SMILES supplied. Pass a list of SMILES strings.")
    if len(smiles) > MAX_BATCH:
        raise ToolError(
            f"Received {len(smiles)} SMILES but the per-call limit is {MAX_BATCH}. "
            "Split the input into smaller batches, or raise HADES_MAX_BATCH.")
    try:
        report = await anyio.to_thread.run_sync(_score_sync, list(smiles))
    except Exception as exc:
        # The SDK logs a ToolError as one line, without its traceback.
        log.exception("HADES scoring failed")
        detail = str(exc) or repr(exc)
        raise ToolError(f"HADES scoring failed: {type(exc).__name__}: {detail}") from exc
    return ScoreReport.model_validate(report)


def _selftest(smiles: list[str]) -> int:
    """Score some molecules and report on stderr. Bypasses MCP entirely."""
    samples = smiles or ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "not-a-molecule"]
    print(f"models: {_model_paths()}", file=sys.stderr)
    report = _score_sync(samples)
    for entry in report["invalid"]:
        print(f"  REJECTED  {entry['smiles']!r}: {entry['reason']}", file=sys.stderr)
    for entry in report["results"]:
        print(f"  {entry['hades_score']:.4f}  {entry['smiles']}", file=sys.stderr)
    return 0 if report["results"] else 1


def main():
    """
    Command-line entry point: serve over stdio (default) or streamable HTTP, or
    run `--selftest`.
    """
    p = argparse.ArgumentParser(description="Serve HADES as an MCP tool (stdio by default).")
    p.add_argument("--selftest", nargs="*", metavar="SMILES",
                   help="score these SMILES (or a few examples) without MCP, then exit")
    p.add_argument("--http", action="store_true",
                   help="serve streamable HTTP at http://HOST:PORT/mcp instead of stdio")
    p.add_argument("--host", help="with --http; default HADES_HOST, else 127.0.0.1")
    p.add_argument("--port", type=int, help="with --http; default HADES_PORT, else 8000")
    args = p.parse_args()

    os.environ.setdefault("TQDM_DISABLE", "1")
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    if args.selftest is not None:
        raise SystemExit(_selftest(args.selftest))
    if args.http:
        # Read the environment only here, so a bad HADES_PORT cannot stop stdio mode.
        host = args.host or os.environ.get("HADES_HOST") or "127.0.0.1"
        port = args.port or int(os.environ.get("HADES_PORT") or 8000)
        log.info("Serving HADES over streamable HTTP on %s:%d/mcp", host, port)
        anyio.run(lambda: mcp.run_streamable_http_async(host=host, port=port))
    else:
        sys.stdout.reconfigure(line_buffering=True)  # see the module docstring
        mcp.run()


if __name__ == "__main__":
    main()
