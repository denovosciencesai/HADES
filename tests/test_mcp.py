"""Checks for the MCP server, `hades_mcp.py`.

Skipped unless the MCP SDK 2.x is installed (`pip install -e ".[mcp]"`).

Two things matter here: every score must come back attached to the molecule it
belongs to, and nothing HADES or its dependencies print may reach stdout, which
on the stdio transport is the JSON-RPC channel itself.
"""

import json
import os
import sys
from pathlib import Path

import pytest

# Gate on the 2.x API: mcp 1.x imports fine but has no mcp.server.mcpserver.
pytest.importorskip("mcp.server.mcpserver", reason='needs the MCP SDK 2.x: pip install -e ".[mcp]"')

import anyio  # noqa: E402
from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

import hades_mcp  # noqa: E402
from test_hades import GOLDEN, TOLERANCE  # noqa: E402 -- one set of recorded scores

REPO = Path(hades_mcp.__file__).resolve().parent
# A bound on each real client session, so a server that answers on the wrong
# stream fails the test instead of hanging it. Generous: the first call loads the
# models, and starting Python from a slow disk takes a while too.
SESSION_SECONDS = 900


@pytest.fixture(autouse=True)
def default_configuration(monkeypatch):
    """Ignore HADES_* settings a developer may have exported for their own client."""
    for name in ("HADES_MODELS_DIR", "HADES_X_TRAIN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(hades_mcp, "REPO_ROOT", REPO)
    monkeypatch.setattr(hades_mcp, "VERIFY", "auto")
    monkeypatch.setattr(hades_mcp, "MAX_BATCH", 100)


def server_env():
    """The environment for a server subprocess: no HADES_* settings, and no
    PYTHONUNBUFFERED, which would hide whether main() line-buffers stdout."""
    return {name: value for name, value in os.environ.items()
            if not name.startswith("HADES_") and name != "PYTHONUNBUFFERED"}


def score(smiles):
    return anyio.run(hades_mcp.hades_score, smiles).model_dump()


def call_over_stdio(params, smiles, errlog=None):
    """List the tools and call hades_score once, in a real client session.

    The server's stderr goes to `sys.__stderr__` by default: it needs a real
    file, and pytest may have swapped `sys.stderr` for one without a descriptor.
    """
    async def session_run():
        with anyio.fail_after(SESSION_SECONDS):
            async with stdio_client(params, errlog=errlog or sys.__stderr__) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    result = await session.call_tool("hades_score", {"smiles": smiles})
                    return listed, result

    listed, result = anyio.run(session_run)
    assert not result.is_error, result.content
    return listed, result.structured_content or json.loads(result.content[0].text)


def test_scores_stay_matched_to_their_inputs():
    """Rejected inputs must not shift the scores of the molecules after them."""
    out = score(["CCO", "NOT_A_SMILES", "  CCCC  ", "", "CCO", "OC(O)[*:1]"])
    assert [r["index"] for r in out["results"]] == [0, 2, 4]
    assert [r["smiles"] for r in out["results"]] == ["CCO", "CCCC", "CCO"]
    assert [r["hades_score"] for r in out["results"]] == pytest.approx(
        [GOLDEN["CCO"], GOLDEN["CCCC"], GOLDEN["CCO"]], abs=TOLERANCE)
    assert [r["index"] for r in out["invalid"]] == [1, 3, 5]
    assert all(r["reason"] for r in out["invalid"])


def test_a_single_molecule_is_scored():
    out = score(["CCO"])
    assert out["invalid"] == []
    assert out["results"][0]["hades_score"] == pytest.approx(GOLDEN["CCO"], abs=TOLERANCE)


def test_salts_are_stripped_and_salt_only_inputs_rejected():
    """A salt whose counter-ion HADES strips scores as its parent; an input that
    stripping leaves empty is rejected.

    Stripped to an empty molecule, a salt-only input would otherwise come back
    with the same imputed score as every other one.
    """
    out = score(["CCO.Cl", "CC(=O)O", "[Na+].[Cl-]"])
    assert [r["smiles"] for r in out["results"]] == ["CCO.Cl"]
    assert out["results"][0]["hades_score"] == pytest.approx(GOLDEN["CCO"], abs=TOLERANCE)
    assert [r["index"] for r in out["invalid"]] == [1, 2]
    assert all("salts" in r["reason"] for r in out["invalid"])


def test_a_failed_standardization_is_reported_not_raised():
    """Neutralizing this anion breaks RDKit's valence rules. HADES would fail the
    whole batch on it, so the server reports it and scores the rest."""
    out = score(["CCO", "C[P-](C)(C)(C)(C)C"])
    assert out["results"][0]["hades_score"] == pytest.approx(GOLDEN["CCO"], abs=TOLERANCE)
    assert [r["index"] for r in out["invalid"]] == [1]
    assert "standardize" in out["invalid"][0]["reason"]


def test_nothing_scoreable_is_reported_not_raised():
    out = score(["NOT_A_SMILES"])
    assert out["results"] == []
    assert out["invalid"][0]["smiles"] == "NOT_A_SMILES"


def test_empty_and_oversized_requests_raise():
    """As ToolError: the one exception whose message the SDK shows the model."""
    with pytest.raises(ToolError, match="No SMILES"):
        score([])
    with pytest.raises(ToolError, match="per-call limit"):
        score(["C"] * (hades_mcp.MAX_BATCH + 1))


def test_stdio_round_trip(tmp_path):
    """A real client session over stdio, through model loading and scoring.

    Started outside the repository, as Claude Code starts it in the user's
    project, so the models must be found without the working directory's help.
    Two valid molecules, so HADES featurizes them in worker processes -- forked
    or spawned while the SDK holds stdout -- rather than in the server process.
    """
    params = StdioServerParameters(command=sys.executable, args=[str(REPO / "hades_mcp.py")],
                                   cwd=str(tmp_path), env=server_env())
    listed, payload = call_over_stdio(params, ["CCO", "NOT_A_SMILES", "CCCC"])
    assert [tool.name for tool in listed.tools] == ["hades_score"]
    # A plain array, not anyOf: some clients drop anyOf and show an untyped argument.
    assert listed.tools[0].input_schema["properties"]["smiles"]["type"] == "array"
    assert [r["hades_score"] for r in payload["results"]] == pytest.approx(
        [GOLDEN["CCO"], GOLDEN["CCCC"]], abs=TOLERANCE)
    assert payload["invalid"][0]["index"] == 1


def test_modern_protocol_round_trip():
    """Claude Code negotiates protocol revision 2026-07-28 through server/discover
    rather than the initialize handshake the other stdio tests use."""
    from mcp.client import Client

    params = StdioServerParameters(command=sys.executable, args=[str(REPO / "hades_mcp.py")],
                                   env=server_env())

    async def session_run():
        with anyio.fail_after(SESSION_SECONDS):
            async with Client(stdio_client(params, errlog=sys.__stderr__)) as client:
                listed = await client.list_tools()
                result = await client.call_tool("hades_score", {"smiles": ["NOT_A_SMILES"]})
                return client.protocol_version, listed, result

    version, listed, result = anyio.run(session_run)
    assert version >= "2026-07-28"
    assert [tool.name for tool in listed.tools] == ["hades_score"]
    assert not result.is_error, result.content
    assert result.structured_content["invalid"][0]["smiles"] == "NOT_A_SMILES"


def test_server_reports_its_version():
    """`serverInfo` carries the HADES version.

    Left unset, the SDK reports an empty string, and a client holding a score
    -- or a model writing one into a summary -- cannot say which HADES
    produced it. The number must also belong to the code that does the
    scoring: the package beside this server, not whatever release happens to
    be installed next to it, which is why `_version()` asks HADES before it
    asks the distribution metadata.
    """
    from HADES import __version__

    assert hades_mcp._version() == __version__
    assert hades_mcp.mcp.version == __version__
    assert hades_mcp.mcp.version, "the SDK's empty-string default reached serverInfo"


def test_unparseable_smiles_is_not_echoed_to_output(capfd):
    """A rejected input must not be written to the console.

    RDKit reports parse failures from C++ and quotes the input back
    ("SMILES Parse Error: Failed parsing SMILES 'NOT_A_SMILES'"). In a terminal
    that is harmless. On a hosted server it is the one path by which a
    submitted structure reaches the console of a service that otherwise
    records nothing about a request, and it is the malformed inputs -- a
    mistyped compound among them -- that get echoed. `capfd` rather than
    `capsys` because the write comes from compiled code, straight to fd 2.
    """
    report = hades_mcp._score_sync(["NOT_A_SMILES"])
    out, err = capfd.readouterr()

    assert report["invalid"][0]["smiles"] == "NOT_A_SMILES", report
    assert "NOT_A_SMILES" not in err, f"the rejected input reached stderr: {err!r}"
    assert "NOT_A_SMILES" not in out, f"the rejected input reached stdout: {out!r}"


def test_stray_output_lands_on_stderr(tmp_path):
    """Output printed during a call must reach stderr, never the JSON-RPC stream.

    Stands in for the scientific stack, which prints both through Python and
    from compiled code writing to fd 1 directly. A print that stayed in the
    stdout buffer would instead be flushed at exit, after the SDK has handed fd 1
    back to the client, and arrive there as an unparseable message.
    """
    noisy_server = (
        "import os, hades_mcp\n"
        "score_sync = hades_mcp._score_sync\n"
        "def noisy(requested):\n"
        "    print('PYTHON-NOISE')\n"
        "    os.write(1, b'FD-NOISE\\n')\n"
        "    return score_sync(requested)\n"
        "hades_mcp._score_sync = noisy\n"
        "hades_mcp.main()\n"
    )
    params = StdioServerParameters(command=sys.executable, args=["-c", noisy_server],
                                   cwd=str(REPO), env=server_env())
    errlog_path = tmp_path / "server-stderr.txt"
    with open(errlog_path, "w") as errlog:
        _, payload = call_over_stdio(params, ["NOT_A_SMILES"], errlog=errlog)
    assert payload["invalid"][0]["smiles"] == "NOT_A_SMILES"
    stderr = errlog_path.read_text()
    assert "PYTHON-NOISE" in stderr
    assert "FD-NOISE" in stderr
