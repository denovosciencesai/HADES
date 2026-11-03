# HADES from Claude and other MCP clients

`hades_mcp.py` serves HADES over the [Model Context Protocol](https://modelcontextprotocol.io)
(MCP), so an AI assistant can score molecules while it works: ask Claude to rank a
set of analogues by HADES score and it calls HADES directly. The server exposes one
tool, `hades_score`, which takes a list of SMILES and returns a score for each
molecule, plus a reason for every input it could not score.

**Contents:** [Install](#install) ·
[Claude Code plugin](#claude-code-plugin) ·
[Other MCP clients](#other-mcp-clients) ·
[What the tool returns](#what-the-tool-returns) ·
[Configuration](#configuration)

## Install

The server runs on your machine, from a Python 3.10 environment that has HADES
and the MCP SDK. The simplest is HADES from PyPI with the `mcp` extra, which
also adds a `hades-mcp` command (on Linux, add the CPU-only torch index from
[Installation](../README.md#with-pip)):

```bash
pip install "hades-druglikeness[mcp]"
```

To run the server without managing an environment at all, see
[Without a clone](#without-a-clone). In the conda environment of a clone
([Installation](../README.md#with-conda-from-a-clone)), add the MCP SDK instead:

```bash
conda activate hades
pip install "mcp==2.3.0"
```

From a checkout, `pip install -e ".[mcp]"` does the same and also installs HADES
itself, with the `hades-mcp` command; it re-applies every pin in
`requirements.txt` too. Install it on the system that runs your MCP client: a
Windows client cannot use the plugin with an environment that lives in WSL (see
[Other MCP clients](#other-mcp-clients)).

## Claude Code plugin

This repository is a [Claude Code plugin](https://code.claude.com/docs/en/plugins)
and its own plugin marketplace, `denovo-hades`. In Claude Code, add the marketplace,
then install the plugin from it:

```
/plugin marketplace add denovosciencesai/HADES
/plugin install hades@denovo-hades
```

To install from a local clone instead of GitHub, add the marketplace by path:
`/plugin marketplace add /path/to/HADES`.

The plugin asks for one setting: the absolute path of the Python interpreter of
the environment from [Install](#install). To print it, run this inside that
environment:

```bash
python -c "import sys; print(sys.executable)"
```

From a shell, the same install is:

```bash
claude plugin marketplace add denovosciencesai/HADES
claude plugin install hades@denovo-hades --config python=/home/you/miniconda3/envs/hades/bin/python
```

A shell install never prompts, so pass the interpreter with `--config`, or set it
later with `/plugin configure hades@denovo-hades`. Write the path out in full:
`~` is not expanded, and a wrong path only shows up when the server fails to
start. On Windows the path looks like
`C:/Users/you/miniconda3/envs/hades/python.exe`. Check the server with `/mcp`;
the tool appears as `mcp__plugin_hades_hades__hades_score`. The plugin was tested
with Claude Code 2.1.299 on Windows.

A GitHub install keeps two copies of the repository, models included: the
marketplace clone and the plugin cache, about 210 MB together. Installing from a
local clone copies that whole directory, untracked files included, so install
from a clean checkout. The plugin is pinned to a version, and third-party
marketplaces do not auto-update by default, so pull a new release with
`claude plugin update hades@denovo-hades`.

The plugin runs one thing: `hades_mcp.py`, with the interpreter you configured,
on your machine. Scoring is local. Before loading the pickled models, the server
checks them against `SHA256SUMS`, as described in
[Model files and integrity](model-integrity.md).

To work on the plugin itself, start Claude Code with `claude --plugin-dir .` from
a checkout and run `/plugin configure hades` once. After editing `hades_mcp.py` or
the `HADES` package, reconnect the server in `/mcp`, or restart Claude Code, so a
new server process picks up the change.

## Other MCP clients

Every MCP client needs one command that starts the server. With HADES installed
from PyPI, that is the full path of `hades-mcp` (`which hades-mcp`, or
`where hades-mcp` on Windows). In Claude Code without the plugin:

```bash
claude mcp add --scope user hades -- "$(which hades-mcp)"
```

`--scope user` makes the tool available in every project, as the plugin does;
without it, it exists only in the directory where you ran the command. In other
clients, use the same path as the command, with no arguments.

From a clone, the command is the `hades` environment's Python, with the path
to `hades_mcp.py` as its argument:

```bash
claude mcp add --scope user hades -- /home/you/miniconda3/envs/hades/bin/python /home/you/HADES/hades_mcp.py
```

In Claude Desktop (`claude_desktop_config.json`), and in clients that use the
same format:

```json
{
  "mcpServers": {
    "hades": {
      "command": "/home/you/miniconda3/envs/hades/bin/python",
      "args": ["/path/to/HADES/hades_mcp.py"]
    }
  }
}
```

On Windows, use the environment's `python.exe` and Windows paths, for example
`C:/Users/you/miniconda3/envs/hades/python.exe` and `C:/path/to/HADES/hades_mcp.py`.

If your client runs on Windows but the `hades` environment lives in WSL, launch
the server through WSL: `"command": "wsl.exe"` with
`"args": ["-d", "Ubuntu", "-e", "/home/you/miniconda3/envs/hades/bin/python", "/home/you/HADES/hades_mcp.py"]`.
The plugin cannot do this, so with Claude Code use `claude mcp add` the same way
(`claude mcp add --scope user hades -- wsl.exe -d Ubuntu -e ...`), or run Claude
Code inside WSL.

### Without a clone

[uv](https://docs.astral.sh/uv/)'s `uvx` installs HADES from PyPI into a cached
environment of its own and runs the server from it. The first run downloads
about 3 GB and can take 15 minutes or more, far longer than MCP clients wait for
a server to start, so run it once in a terminal before registering it:

```bash
uvx --python 3.10 --from "hades-druglikeness[mcp]" hades-mcp --selftest
```

It prints a score for ethanol (`0.0178  CCO`). Later runs reuse the cached
environment and start in seconds.

On Linux, PyPI's torch brings several GB of CUDA libraries that HADES does not
use. PyTorch's CPU-only index avoids them, with identical scores. uv needs
`--index-strategy unsafe-best-match` to pick each package from whichever index
has the pinned version, as pip does; without it, uv looks for every package on
the first index that lists it. Use the same options in the commands below:

```bash
uvx --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
    --python 3.10 --from "hades-druglikeness[mcp]" hades-mcp --selftest
```

Then register the same command, in Claude Code:

```bash
claude mcp add --scope user hades -- uvx --python 3.10 --from "hades-druglikeness[mcp]" hades-mcp
```

or in `claude_desktop_config.json` and clients that use its format:

```json
{
  "mcpServers": {
    "hades": {
      "command": "uvx",
      "args": ["--python", "3.10", "--from", "hades-druglikeness[mcp]", "hades-mcp"]
    }
  }
}
```

If the client cannot find `uvx`, give its full path (`which uvx`, or `where uvx`
on Windows).

## What the tool returns

```json
{
  "results": [{"index": 0, "smiles": "CCO", "hades_score": 0.0178}],
  "invalid": [{"index": 1, "smiles": "NOT_A_SMILES", "reason": "RDKit cannot parse it"}]
}
```

`index` is each entry's position in the input list. A call takes up to 100 SMILES,
which keeps a full result well inside the output limits MCP clients apply. The
models load on the first call, which takes a while; later calls are fast.

Pass each drug as its parent compound (free acid or base). Counter-ions on RDKit's
default salt list (Na+, K+, Cl-, mesylate, tartrate and other common ones) are
stripped before scoring, but others, such as citrate, besylate or succinate, are
scored as part of the molecule. Inputs that salt stripping leaves empty, such as
sodium chloride or water, are reported in `invalid`.

## Configuration

The server reads these environment variables, all optional. Where to set them
depends on the client:

- **Claude Code**, plugin or `claude mcp add`: the server inherits the environment
  Claude Code was started from, so export them first. With `claude mcp add` you
  can also pass them with `-e NAME=value`.
- **Claude Desktop and most other clients** pass only a few variables to the
  servers they start. Add an `env` block to the server entry:
  `"env": {"HADES_MODELS_DIR": "/abs/path/run/models", "HADES_X_TRAIN": "/abs/path/run/X_train.parquet"}`.
- **Through `wsl.exe`**, Windows variables reach WSL only when listed in
  `WSLENV`, so set them on the Linux side instead, with `env` before the command:
  `"args": ["-d", "Ubuntu", "-e", "env", "HADES_MODELS_DIR=/home/you/run/models", "/home/you/miniconda3/envs/hades/bin/python", "/home/you/HADES/hades_mcp.py"]`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `HADES_MODELS_DIR` | the shipped models | Score with an ensemble you trained ([Train your own model](../README.md#train-your-own-model)) |
| `HADES_X_TRAIN` | the descriptor spec that goes with the shipped models | The `X_train.parquet` that goes with `HADES_MODELS_DIR` |
| `HADES_VERIFY` | `auto` | Checksum check before loading the models: `auto`, `require` or `off` |
| `HADES_MAX_BATCH` | `100` | Maximum SMILES per call |
| `HADES_REPO` | the directory of `hades_mcp.py` | The HADES checkout to import HADES and its shipped models from |

To check the server without any client, score a few molecules directly:

```bash
python hades_mcp.py --selftest CCO c1ccccc1
```
