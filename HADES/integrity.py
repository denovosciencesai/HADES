"""SHA-256 manifests for HADES model directories.

The trained models are Python pickles, and `joblib.load` executes code from the
file it reads -- so the bytes have to come from a source you trust. A manifest
does not make an untrusted pickle safe. What it does is detect tampering and
corruption, and pin exactly which bytes produced a given set of scores. A
mismatch is a reason to stop, not a reason to fetch the file again.

The manifest is written in the standard `sha256sum` format, so a model
directory can also be checked without Python:

    cd HADES_saved_models && sha256sum -c SHA256SUMS
"""

import glob
import hashlib
import os
import sys

MANIFEST_NAME = "SHA256SUMS"
VERIFY_MODES = ("auto", "require", "off")


class ChecksumError(RuntimeError):
    """A model file's digest does not match the manifest, or the manifest is unusable."""


def _default_warn(message):
    # `HADES.quiet_noise` sets warnings to "ignore" process-wide at import time,
    # so the warnings module cannot be used to say anything here.
    print(f"WARNING: {message}", file=sys.stderr, flush=True)


def sha256_file(path, chunk_size=1 << 20):
    """
    Computes the SHA-256 digest of a file, reading it in chunks.
    Args:
        path (str): File to hash.
        chunk_size (int): Bytes per read. Defaults to 1 MiB, so a 64 MB model
            never has to be held in memory.
    Returns:
        str: The lowercase hex digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(manifest_path):
    """
    Parses a `sha256sum`-style manifest.
    Args:
        manifest_path (str): Path to the manifest file.
    Raises:
        ChecksumError: If a line is not a digest/filename pair, or the manifest
            lists no files.
    Returns:
        dict[str, str]: Filename -> expected lowercase hex digest, in the order
            the manifest lists them.
    """
    entries = {}
    with open(manifest_path) as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            digest, _, name = line.partition(" ")
            name = name.lstrip(" *")  # sha256sum marks binary mode with a leading '*'
            if len(digest) != 64 or not name:
                raise ChecksumError(f"{manifest_path}:{lineno}: not a sha256sum line: {raw!r}")
            entries[name] = digest.lower()
    if not entries:
        raise ChecksumError(f"{manifest_path} lists no files")
    return entries


def write_manifest(models_dir, filenames=None):
    """
    Writes a `SHA256SUMS` manifest for a directory of model pickles.

    Use this after training your own ensemble, so `HADES_score` can verify it
    the same way it verifies the models shipped with the repository.

    Args:
        models_dir (str): Directory holding the `.pkl` files.
        filenames (list[str], optional): Filenames to record, relative to
            `models_dir`. Defaults to every `.pkl` in the directory, sorted.
    Returns:
        str: Path to the manifest that was written.
    """
    models_dir = os.fspath(models_dir)
    if filenames is None:
        filenames = sorted(n for n in os.listdir(models_dir) if n.endswith(".pkl"))
    manifest_path = os.path.join(models_dir, MANIFEST_NAME)
    with open(manifest_path, "w") as f:
        for name in filenames:
            f.write(f"{sha256_file(os.path.join(models_dir, name))}  {name}\n")
    return manifest_path


def verify_model_dir(models_dir, mode="auto", on_warning=None):
    """
    Checks the `.pkl` files in `models_dir` against its `SHA256SUMS` manifest.

    Which files get loaded is not decided here -- the caller still globs the
    directory. This only reports whether the bytes are the expected ones.

    Args:
        models_dir (str): Directory containing the trained `.pkl` files.
        mode (str): "auto" verifies against the manifest when one is present and
            warns on stderr when it is not; "require" treats a missing manifest,
            and any `.pkl` the manifest does not list, as errors; "off" skips
            the check entirely.
        on_warning (callable, optional): Receives warning messages. Defaults to
            printing to stderr.
    Raises:
        ValueError: If `mode` is not one of "auto", "require" or "off".
        ChecksumError: If a digest does not match, a manifest entry is missing
            from disk, or `mode="require"` and the manifest is missing or
            incomplete.
    Returns:
        dict[str, str] or None: The manifest entries that were checked, or None
            when no check was performed.
    """
    if mode not in VERIFY_MODES:
        raise ValueError(f"verify must be one of {VERIFY_MODES}, got {mode!r}")
    if mode == "off":
        return None

    models_dir = os.fspath(models_dir)
    warn = on_warning or _default_warn
    manifest_path = os.path.join(models_dir, MANIFEST_NAME)

    if not os.path.isfile(manifest_path):
        if mode == "require":
            raise ChecksumError(f"verify='require' but no {MANIFEST_NAME} in {models_dir}")
        warn(f"no {MANIFEST_NAME} in {models_dir} -- loading unverified pickles. Only load "
             f"models from a source you trust. To record digests for a directory you "
             f"produced yourself: HADES.integrity.write_manifest({models_dir!r})")
        return None

    entries = read_manifest(manifest_path)
    on_disk = sorted(os.path.basename(p) for p in glob.glob(os.path.join(models_dir, "*.pkl")))

    mismatched, unlisted = [], []
    for name in on_disk:
        if name not in entries:
            unlisted.append(name)
            continue
        actual = sha256_file(os.path.join(models_dir, name))
        if actual != entries[name]:
            mismatched.append(f"  {name}: expected {entries[name]}, got {actual}")
    missing = [name for name in entries if name not in set(on_disk)]

    problems = []
    if mismatched:
        problems.append("checksum mismatch:\n" + "\n".join(mismatched))
    if missing:
        problems.append(f"listed in {MANIFEST_NAME} but not on disk: {missing}")
    if unlisted and mode == "require":
        problems.append(f"present but not listed in {MANIFEST_NAME}: {unlisted}")
    if problems:
        raise ChecksumError(
            f"{manifest_path} does not describe {models_dir}:\n" + "\n".join(problems) +
            "\nThese files are pickles; do not load them until you know why they changed.")

    if unlisted:
        warn(f"{MANIFEST_NAME} does not list {unlisted} in {models_dir}; "
             f"these will still be loaded, unverified.")
    return entries
