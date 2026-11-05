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
import importlib.metadata
import inspect
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

# For every HADES_* variable, an empty value means unset.
REPO_ROOT = Path(os.environ.get("HADES_REPO") or Path(__file__).resolve().parent)
VERIFY = os.environ.get("HADES_VERIFY") or "auto"
MAX_BATCH = int(os.environ.get("HADES_MAX_BATCH") or 100)


def _importable():
    """Put the repository on sys.path, so HADES_CALCULATOR and HADES import from it."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _version() -> str:
    """The HADES version, for the `serverInfo` a client reads on initialize.

    Left unset, the SDK reports an empty string, and nothing downstream can say
    which HADES produced a score -- including a model writing the number into a
    summary, and a reviewer asked to reproduce it.

    Ask the package before the distribution metadata, because they can
    disagree and only one of them is the truth. `_importable()` puts REPO_ROOT
    at the front of `sys.path`, so `HADES.__version__` is the version of the
    code and models this server will actually score with: the clone's, when
    running from a clone that sits beside a different installed release, and
    the installed one otherwise. `importlib.metadata` would answer with
    whatever is installed either way, which is the wrong number in exactly the
    case where the two differ. It stays as the fallback for an installation
    whose package cannot be imported.

    Importing HADES here is cheap: its re-exports are lazy, so this does not
    pull torch in at startup.
    """
    try:
        _importable()
        from HADES import __version__  # noqa: PLC0415 -- deliberately lazy
        return __version__
    except Exception:
        pass
    try:
        return importlib.metadata.version("hades-druglikeness")
    except importlib.metadata.PackageNotFoundError:
        log.warning("Cannot determine the HADES version; reporting it as 'unknown'.")
        return "unknown"


mcp = MCPServer(
    name="hades",
    version=_version(),
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


def _model_paths(scorer: type) -> dict[str, str]:
    """The model and descriptor paths to pass to `scorer`, the HADES_score class.

    The server runs in the client's working directory -- the user's project,
    not this repository. Where HADES_score's default paths are relative (models
    kept in the repository), resolve them against REPO_ROOT instead of that
    directory; where they are absolute or unset (models installed with the
    package), keep them. Either way the server loads the models `hades-score`
    loads from the repository root. HADES_MODELS_DIR and HADES_X_TRAIN override
    either path.
    """
    defaults = inspect.signature(scorer).parameters
    paths = {}
    for name in ("models_dir", "x_train_parquet"):
        default = defaults[name].default
        if isinstance(default, str) and not os.path.isabs(default):
            paths[name] = str(REPO_ROOT / default)
    if os.environ.get("HADES_MODELS_DIR"):
        paths["models_dir"] = os.environ["HADES_MODELS_DIR"]
    if os.environ.get("HADES_X_TRAIN"):
        paths["x_train_parquet"] = os.environ["HADES_X_TRAIN"]
    return paths


def _load_hades() -> Any:
    """Build the HADES scorer. Caller must hold _model_lock."""
    _importable()
    from HADES_CALCULATOR import HADES_score  # noqa: PLC0415 -- deliberately lazy

    paths = _model_paths(HADES_score)
    log.info("Loading HADES (first call only, this is slow): %s",
             paths or "HADES_score's default paths")
    try:
        scorer = HADES_score(quiet=True, verify=VERIFY, **paths)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"{exc}. Set HADES_REPO to the HADES repository root, or HADES_MODELS_DIR "
            "and HADES_X_TRAIN to a model you trained.") from exc
    log.info("HADES ready: %d ensemble members.", len(scorer.models))
    return scorer


_rdkit_silenced = False


def _silence_rdkit() -> None:
    """Stop RDKit writing the molecules it cannot parse to stderr.

    RDKit reports parse failures from C++ through its `rdApp.*` loggers, which
    `HADES.quiet_noise` does not reach -- that silences Python loggers,
    warnings and tqdm. The message quotes the input back:

        SMILES Parse Error: Failed parsing SMILES 'NOT_A_SMILES'

    Harmless in a terminal, where the structure is the user's own and already
    on their screen. On a server it is the one path by which a submitted
    structure reaches the console, in a service that otherwise records nothing
    about a request -- and it is the malformed inputs, a mistyped compound
    among them, that get echoed. The caller is told why the input failed
    through `invalid` either way, so silencing the duplicate loses nothing.

    Idempotent: `DisableLog` is cheap, but importing rdkit is not, and this
    runs on every scoring call.
    """
    global _rdkit_silenced
    if _rdkit_silenced:
        return
    from rdkit import RDLogger  # noqa: PLC0415 -- deliberately lazy

    RDLogger.DisableLog("rdApp.*")
    _rdkit_silenced = True


def _standardization_problem(smiles: str) -> str | None:
    """Why the molecule HADES would score, once standardized, cannot be scored."""
    from rdkit import Chem  # noqa: PLC0415 -- deliberately lazy
    from HADES.featurize import Featurizer  # noqa: PLC0415

    try:
        standardized = Featurizer.standartize_mol(Chem.MolFromSmiles(smiles))
    except Exception as exc:  # for one, RDKit valence errors on neutralization
        return f"HADES cannot standardize it: {exc}"
    if standardized.GetNumAtoms() == 0:
        return "nothing is left once salts and counter-ions are stripped"
    return None


def _score_sync(requested: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Validate, featurize and score. Runs in a worker thread; serialized by _model_lock.

    Inputs are checked with the same `unscoreable_reason` gate the featurizer
    applies, so everything that passes it gets a score. `score()` returns the
    scored SMILES without their input positions; checking that they come back
    exactly as sent is what keeps each score attached to its molecule.

    The gate looks at the molecule as written, but HADES scores it after
    stripping the counter-ions on RDKit's default salt list and neutralizing
    charges. An input made only of those (NaCl, acetic acid, water) is stripped
    to an empty molecule, whose descriptors -- constants, and missing values
    filled with training means -- give every such input the same score. Those
    inputs are rejected here as well, as are inputs the standardization itself
    fails on, which would otherwise fail the whole batch.
    """
    global _hades
    with _model_lock:
        _importable()
        _silence_rdkit()
        from HADES.featurize import unscoreable_reason  # noqa: PLC0415 -- deliberately lazy

        valid: list[tuple[int, str]] = []
        invalid: list[dict[str, Any]] = []
        for index, raw in enumerate(requested):
            candidate = raw.strip()
            reason = unscoreable_reason(candidate) or _standardization_problem(candidate)
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
        f"{MAX_BATCH}. Pass each drug as its parent compound (free acid or "
        "base): HADES strips common counter-ions such as Na+, K+, Cl- or "
        "mesylate, but not others such as citrate, besylate or succinate, nor "
        "the second parent of a 2:1 salt, and those change the score. Inputs "
        "that cannot be scored (for example unparseable, empty, fragments with "
        "attachment points, or nothing left once salts are stripped, such as "
        "NaCl or water) are reported in 'invalid' with a reason rather than "
        "failing the whole call. Each entry carries 'index', its position in "
        "the input. A low score for chemistry far outside the training "
        "distribution is not evidence that a molecule is not drug-like."
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


# An approved oral drug, something nobody would call one, and a string that is
# not a molecule at all: enough to show the score moving across its range and
# the rejection path working. The previous default scored three near-zeros,
# which demonstrated only half of that.
SELFTEST_SAMPLES = [
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
    "CCO",                         # ethanol
    "not-a-molecule",
]


def _selftest(smiles: list[str]) -> int:
    """Score some molecules and report on stderr. Bypasses MCP entirely."""
    samples = smiles or list(SELFTEST_SAMPLES)
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
        host = args.host if args.host is not None else os.environ.get("HADES_HOST") or "127.0.0.1"
        port = args.port if args.port is not None else int(os.environ.get("HADES_PORT") or 8000)
        log.info("Serving HADES over streamable HTTP on %s:%d/mcp", host, port)
        anyio.run(lambda: mcp.run_streamable_http_async(host=host, port=port))
    else:
        sys.stdout.reconfigure(line_buffering=True)  # see the module docstring
        mcp.run()


if __name__ == "__main__":
    main()
