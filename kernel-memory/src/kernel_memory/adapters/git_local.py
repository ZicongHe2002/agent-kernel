"""Read-only inspection of a local Git repository (specification sections 7, 8, 19).

Public API
----------
``LocalGitRepo(path, *, timeout=120.0)``
    Wraps one repository. The constructor runs ``git rev-parse --show-toplevel`` (falling
    back to ``--is-bare-repository`` for bare repositories) and raises ``InputError``
    (code ``NOT_A_GIT_REPOSITORY``) when ``path`` is not a Git work tree or repository.
    Every command is executed as an argument array with ``subprocess.run(..., shell=False,
    cwd=path, capture_output=True, check=False, timeout=...)``; nothing is ever passed
    through a shell and the Git configuration is never modified (``GIT_OPTIONAL_LOCKS=0``
    keeps ``status`` from touching the index, ``GIT_TERMINAL_PROMPT=0`` forbids prompts).

    * ``object_format() -> "sha1" | "sha256"``
    * ``rev_parse(ref) -> GitOid``: full object id or ``MissingReferenceError`` with code
      ``UNKNOWN_REVISION`` (unknown) / ``AMBIGUOUS_REVISION`` (Git reports ambiguity).
      Short SHAs are resolved by Git to the full id or fail; they are never padded.
    * ``commit_info(oid) -> CommitInfo`` (parents via ``git rev-list --parents -n 1``,
      tree/author/message via ``git cat-file commit``). Message text is data.
    * ``rev_list(head, exclude) -> list[GitOid]`` (``git rev-list --topo-order --reverse
      HEAD --not EXCLUDE...``), oldest first.
    * ``is_shallow()``, ``merge_base(a, b) -> GitOid | None``, ``has_object(oid) -> bool`` (full
      object ids only; an abbreviation answers ``False`` rather than being resolved),
      ``tree_oid(commit)``, ``head_oid()``.
    * ``diff_patch(base, head) -> bytes`` (binary-safe, config-independent flags) and
      ``diff_numstat(base, head) -> list[{"path", "added", "deleted"}]`` (``None`` counts
      for binary files, never ``0``).
    * ``is_dirty()`` (``git status --porcelain --untracked-files=normal``) and
      ``dirty_patch_digest() -> str | None`` (``sha256:`` digest over ``git diff HEAD`` plus
      the digests of untracked files; ``None`` when clean).
    * ``source_manifest(commit) -> list[(relative path, "sha256:<hex>")]`` from
      ``git archive --format=tar`` parsed in memory, and ``source_digest(commit)`` =
      ``hashing.source_digest(manifest)``.

Guarantees: every identifier handed to Git is validated first (no leading ``-``, no
control characters, hex checked for object ids), so untrusted refs cannot become options.
Failures of the ``git`` binary itself surface as ``PrerequisiteMissingError``
(``GIT_NOT_AVAILABLE``) or ``ExecutionInfrastructureError`` (``GIT_TIMEOUT`` /
``GIT_COMMAND_FAILED``); nothing here writes to the repository.
"""
from __future__ import annotations

import io
import os
import re
import subprocess
import tarfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

from ..domain.errors import (
    ExecutionInfrastructureError,
    InputError,
    MissingReferenceError,
    PrerequisiteMissingError,
)
from ..domain.hashing import jcs_digest, sha256_bytes
from ..domain.hashing import source_digest as compute_source_digest
from ..domain.ids import resolve_inside
from ..domain.models import GitOid

_HEX_RE = re.compile(r"^[0-9a-f]+$")
_OID_LENGTHS = {"sha1": 40, "sha256": 64}
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_AUTHOR_RE = re.compile(r"^(?P<name>.*?) <(?P<email>[^>]*)> (?P<ts>\d+) (?P<tz>[+-]\d{4})$")
_STDERR_LIMIT = 2000


@dataclass(frozen=True)
class CommitInfo:
    oid: GitOid
    parents: list[GitOid]
    tree: GitOid
    subject: str
    message: str
    author_date: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "oid": {"algorithm": self.oid.algorithm, "hex": self.oid.hex},
            "parents": [{"algorithm": p.algorithm, "hex": p.hex} for p in self.parents],
            "tree": {"algorithm": self.tree.algorithm, "hex": self.tree.hex},
            "subject": self.subject,
            "message": self.message,
            "author_date": self.author_date,
        }


def validate_ref(ref: str, *, what: str = "revision") -> str:
    """Reject values that could be interpreted as Git options or contain control bytes."""
    if not isinstance(ref, str) or not ref:
        raise InputError(f"{what} must be a non-empty string", code="INVALID_REVISION")
    if ref.startswith("-"):
        raise InputError(f"{what} may not start with '-': {ref!r}", code="INVALID_REVISION")
    if _CONTROL_RE.search(ref) or any(ch.isspace() for ch in ref):
        raise InputError(f"{what} contains whitespace or control characters: {ref!r}", code="INVALID_REVISION")
    if len(ref) > 512:
        raise InputError(f"{what} is too long", code="INVALID_REVISION")
    return ref


def algorithm_for_hex(hex_value: str) -> str:
    """Map a full hexadecimal object id to its algorithm, refusing anything else."""
    if not isinstance(hex_value, str) or not _HEX_RE.match(hex_value):
        raise InputError(f"object id is not lowercase hexadecimal: {hex_value!r}", code="INVALID_OID")
    for algorithm, length in _OID_LENGTHS.items():
        if len(hex_value) == length:
            return algorithm
    raise InputError(
        f"object id has {len(hex_value)} hex digits; full ids are 40 (sha1) or 64 (sha256) digits and are never padded",
        code="INVALID_OID",
    )


def oid_from_hex(hex_value: str) -> GitOid:
    return GitOid(algorithm=algorithm_for_hex(hex_value), hex=hex_value)


def _oid_arg(value: GitOid | str) -> str:
    hex_value = value.hex if isinstance(value, GitOid) else value
    if isinstance(value, GitOid):
        algorithm_for_hex(hex_value)
        return hex_value
    return validate_ref(hex_value)


class LocalGitRepo:
    def __init__(self, path: Path | str, *, timeout: float = 120.0) -> None:
        self.path = Path(path)
        self.timeout = float(timeout)
        if not self.path.is_dir():
            raise InputError(f"repository path is not a directory: {self.path}", code="NOT_A_GIT_REPOSITORY")
        self._object_format: str | None = None
        self.toplevel: Path | None = None
        self.bare = False
        top = self._run(["rev-parse", "--show-toplevel"])
        if top.returncode == 0:
            self.toplevel = Path(top.stdout.decode("utf-8", errors="surrogateescape").strip())
            return
        bare = self._run(["rev-parse", "--is-bare-repository"])
        if bare.returncode == 0 and bare.stdout.decode("utf-8", errors="replace").strip() == "true":
            self.bare = True
            return
        raise InputError(
            f"{self.path} is not a git work tree or repository",
            code="NOT_A_GIT_REPOSITORY",
            details={"path": str(self.path), "git_stderr": _tail(top.stderr)},
        )

    # ------------------------------------------------------------------ process plumbing
    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["GIT_OPTIONAL_LOCKS"] = "0"
        env["LC_ALL"] = "C"
        env.pop("GIT_DIR", None)
        env.pop("GIT_WORK_TREE", None)
        return env

    def _run(self, args: Sequence[str], *, input_bytes: bytes | None = None) -> subprocess.CompletedProcess:
        argv = ["git", *args]
        try:
            return subprocess.run(  # noqa: S603 - argument array, shell=False, validated inputs
                argv,
                cwd=str(self.path),
                capture_output=True,
                check=False,
                timeout=self.timeout,
                env=self._env(),
                input=input_bytes,
            )
        except FileNotFoundError as exc:
            raise PrerequisiteMissingError("the git executable is not available on PATH", code="GIT_NOT_AVAILABLE") from exc
        except subprocess.TimeoutExpired as exc:
            raise ExecutionInfrastructureError(
                f"git {' '.join(args[:2])} exceeded the {self.timeout:g}s timeout",
                code="GIT_TIMEOUT",
                details={"args": list(args)},
            ) from exc

    def _run_ok(self, args: Sequence[str], *, code: str = "GIT_COMMAND_FAILED") -> bytes:
        proc = self._run(args)
        if proc.returncode != 0:
            raise ExecutionInfrastructureError(
                f"git {' '.join(args)} failed with exit status {proc.returncode}: {_tail(proc.stderr)}",
                code=code,
                details={"args": list(args), "returncode": proc.returncode, "stderr": _tail(proc.stderr)},
            )
        return proc.stdout

    def _text(self, args: Sequence[str]) -> str:
        return self._run_ok(args).decode("utf-8", errors="replace").strip()

    # ------------------------------------------------------------------ identity
    def object_format(self) -> str:
        if self._object_format is None:
            proc = self._run(["rev-parse", "--show-object-format"])
            fmt = proc.stdout.decode("utf-8", errors="replace").strip() if proc.returncode == 0 else "sha1"
            if fmt not in _OID_LENGTHS:
                raise ExecutionInfrastructureError(f"unsupported git object format {fmt!r}", code="UNSUPPORTED_OBJECT_FORMAT")
            self._object_format = fmt
        return self._object_format

    def _oid(self, hex_value: str) -> GitOid:
        fmt = self.object_format()
        if not _HEX_RE.match(hex_value) or len(hex_value) != _OID_LENGTHS[fmt]:
            raise ExecutionInfrastructureError(
                f"git returned an unexpected object id {hex_value!r} for object format {fmt}", code="GIT_COMMAND_FAILED"
            )
        return GitOid(algorithm=fmt, hex=hex_value)

    def rev_parse(self, ref: str) -> GitOid:
        """Resolve any revision expression (full/short sha, branch, tag, HEAD) to a full commit id."""
        validate_ref(ref)
        proc = self._run(["rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"])
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace")
            if "ambiguous" in stderr.lower():
                raise MissingReferenceError(
                    f"revision {ref!r} is ambiguous in {self.path}; supply more characters or the full object id",
                    code="AMBIGUOUS_REVISION",
                    details={"ref": ref, "git_stderr": _tail(proc.stderr)},
                )
            raise MissingReferenceError(
                f"revision {ref!r} is unknown in {self.path}",
                code="UNKNOWN_REVISION",
                details={"ref": ref, "git_stderr": _tail(proc.stderr)},
            )
        return self._oid(proc.stdout.decode("utf-8", errors="replace").strip())

    def head_oid(self) -> GitOid:
        return self.rev_parse("HEAD")

    def has_object(self, oid: GitOid | str) -> bool:
        """True when the *full* object id is stored. Abbreviations are never resolved here (T31)."""
        try:
            arg = _oid_arg(oid)
            algorithm = algorithm_for_hex(arg)
        except InputError:
            return False
        if algorithm != self.object_format():
            return False
        proc = self._run(["cat-file", "-e", arg])
        return proc.returncode == 0

    def tree_oid(self, commit: GitOid | str) -> GitOid:
        arg = _oid_arg(commit)
        proc = self._run(["rev-parse", "--verify", "--end-of-options", f"{arg}^{{tree}}"])
        if proc.returncode != 0:
            raise MissingReferenceError(
                f"cannot resolve the tree of {arg!r}", code="UNKNOWN_REVISION", details={"ref": arg, "git_stderr": _tail(proc.stderr)}
            )
        return self._oid(proc.stdout.decode("utf-8", errors="replace").strip())

    def is_shallow(self) -> bool:
        return self._text(["rev-parse", "--is-shallow-repository"]) == "true"

    # ------------------------------------------------------------------ history
    def commit_info(self, oid: GitOid | str) -> CommitInfo:
        arg = _oid_arg(oid)
        full = self.rev_parse(arg)
        parents_line = self._text(["rev-list", "--parents", "-n", "1", full.hex])
        fields = parents_line.split()
        if not fields or fields[0] != full.hex:
            raise ExecutionInfrastructureError(f"unexpected rev-list output for {full.hex}: {parents_line!r}", code="GIT_COMMAND_FAILED")
        parents = [self._oid(p) for p in fields[1:]]
        raw = self._run_ok(["cat-file", "commit", full.hex])
        headers, message = _split_commit_object(raw)
        tree_hex = headers.get("tree", [""])[0]
        if not tree_hex:
            raise ExecutionInfrastructureError(f"commit {full.hex} has no tree header", code="GIT_COMMAND_FAILED")
        author_date = _author_date(headers.get("author", [""])[0])
        subject = message.split("\n", 1)[0].strip() if message else ""
        return CommitInfo(oid=full, parents=parents, tree=self._oid(tree_hex), subject=subject, message=message, author_date=author_date)

    def rev_list(self, head: GitOid | str, exclude: Sequence[GitOid | str] = ()) -> list[GitOid]:
        head_arg = _oid_arg(head)
        args = ["rev-list", "--topo-order", "--reverse", head_arg]
        excludes = [_oid_arg(e) for e in exclude]
        if excludes:
            args.append("--not")
            args.extend(excludes)
        proc = self._run(args)
        if proc.returncode != 0:
            raise MissingReferenceError(
                f"git rev-list failed for {head_arg!r}: {_tail(proc.stderr)}",
                code="UNKNOWN_REVISION",
                details={"head": head_arg, "exclude": excludes, "git_stderr": _tail(proc.stderr)},
            )
        return [self._oid(line.strip()) for line in proc.stdout.decode("utf-8", errors="replace").splitlines() if line.strip()]

    def merge_base(self, a: GitOid | str, b: GitOid | str) -> GitOid | None:
        proc = self._run(["merge-base", _oid_arg(a), _oid_arg(b)])
        if proc.returncode == 1:
            return None
        if proc.returncode != 0:
            raise MissingReferenceError(
                f"git merge-base failed: {_tail(proc.stderr)}", code="UNKNOWN_REVISION", details={"git_stderr": _tail(proc.stderr)}
            )
        return self._oid(proc.stdout.decode("utf-8", errors="replace").strip())

    # ------------------------------------------------------------------ diffs
    _DIFF_FLAGS = (
        "--no-color",
        "--no-ext-diff",
        "--no-renames",
        "--full-index",
        "--diff-algorithm=myers",
        "--src-prefix=a/",
        "--dst-prefix=b/",
    )

    def diff_patch(self, base: GitOid | str, head: GitOid | str) -> bytes:
        """Binary-safe unified diff between two commits (bytes, never decoded)."""
        base_arg, head_arg = _oid_arg(base), _oid_arg(head)
        proc = self._run(["diff", *self._DIFF_FLAGS, "--binary", base_arg, head_arg, "--"])
        if proc.returncode not in (0, 1):
            raise MissingReferenceError(
                f"git diff {base_arg} {head_arg} failed: {_tail(proc.stderr)}",
                code="UNKNOWN_REVISION",
                details={"base": base_arg, "head": head_arg, "git_stderr": _tail(proc.stderr)},
            )
        return proc.stdout

    def diff_numstat(self, base: GitOid | str, head: GitOid | str) -> list[dict[str, Any]]:
        base_arg, head_arg = _oid_arg(base), _oid_arg(head)
        proc = self._run(["diff", "--numstat", "--no-renames", "--no-ext-diff", "-z", base_arg, head_arg, "--"])
        if proc.returncode not in (0, 1):
            raise MissingReferenceError(
                f"git diff --numstat {base_arg} {head_arg} failed: {_tail(proc.stderr)}",
                code="UNKNOWN_REVISION",
                details={"base": base_arg, "head": head_arg, "git_stderr": _tail(proc.stderr)},
            )
        out: list[dict[str, Any]] = []
        for chunk in proc.stdout.split(b"\x00"):
            if not chunk:
                continue
            parts = chunk.split(b"\t", 2)
            if len(parts) != 3:
                continue
            added_raw, deleted_raw, path_raw = parts
            path = path_raw.decode("utf-8", errors="surrogateescape")
            added = None if added_raw == b"-" else int(added_raw)
            deleted = None if deleted_raw == b"-" else int(deleted_raw)
            out.append({"path": path, "added": added, "deleted": deleted})
        return out

    # ------------------------------------------------------------------ work tree state
    def is_dirty(self) -> bool:
        if self.bare:
            return False
        out = self._run_ok(["status", "--porcelain", "--untracked-files=normal"])
        return bool(out.strip())

    def dirty_patch_digest(self) -> str | None:
        """Digest of tracked modifications (``git diff HEAD``) plus untracked file contents; None when clean."""
        if not self.is_dirty():
            return None
        proc = self._run(["diff", *self._DIFF_FLAGS, "--binary", "HEAD", "--"])
        if proc.returncode not in (0, 1):
            raise ExecutionInfrastructureError(f"git diff HEAD failed: {_tail(proc.stderr)}", code="GIT_COMMAND_FAILED")
        tracked = sha256_bytes(proc.stdout)
        untracked: list[list[str]] = []
        listing = self._run_ok(["ls-files", "--others", "--exclude-standard", "-z"])
        root = self.toplevel or self.path
        for raw in listing.split(b"\x00"):
            if not raw:
                continue
            rel = raw.decode("utf-8", errors="surrogateescape")
            target = resolve_inside(root, rel)
            if target.is_file():
                untracked.append([rel, sha256_bytes(target.read_bytes())])
        untracked.sort()
        return jcs_digest({"dirty_patch_version": 1, "tracked_diff": tracked, "untracked": untracked})

    # ------------------------------------------------------------------ source manifest
    def source_manifest(self, commit: GitOid | str) -> list[tuple[str, str]]:
        """(relative path, sha256 digest of bytes) for every file in the commit's tree, sorted by path."""
        arg = _oid_arg(commit)
        full = self.rev_parse(arg)
        data = self._run_ok(["archive", "--format=tar", full.hex])
        entries: list[tuple[str, str]] = []
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tar:
                for member in tar.getmembers():
                    name = member.name
                    if name.startswith("./"):
                        name = name[2:]
                    if not name or name == "pax_global_header":
                        continue
                    if member.isfile():
                        handle = tar.extractfile(member)
                        content = handle.read() if handle is not None else b""
                        entries.append((name, sha256_bytes(content)))
                    elif member.issym():
                        entries.append((name, sha256_bytes(b"symlink:" + member.linkname.encode("utf-8", errors="surrogateescape"))))
        except tarfile.TarError as exc:
            raise ExecutionInfrastructureError(f"cannot parse git archive output for {full.hex}: {exc}", code="GIT_COMMAND_FAILED") from exc
        entries.sort()
        return entries

    def source_digest(self, commit: GitOid | str) -> str:
        return compute_source_digest(self.source_manifest(commit))


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _tail(data: bytes | None) -> str:
    if not data:
        return ""
    text = data.decode("utf-8", errors="replace").strip()
    return text[-_STDERR_LIMIT:]


def _split_commit_object(raw: bytes) -> tuple[dict[str, list[str]], str]:
    """Parse a raw commit object into headers (multi-valued) and the message text."""
    text = raw.decode("utf-8", errors="replace")
    head, sep, message = text.partition("\n\n")
    if not sep:
        head, message = text, ""
    headers: dict[str, list[str]] = {}
    last_key: str | None = None
    for line in head.split("\n"):
        if not line:
            continue
        if line.startswith(" ") and last_key is not None:
            headers[last_key][-1] += "\n" + line[1:]
            continue
        key, _, value = line.partition(" ")
        headers.setdefault(key, []).append(value)
        last_key = key
    return headers, message


def _author_date(author_line: str) -> str:
    match = _AUTHOR_RE.match(author_line or "")
    if not match:
        return ""
    ts = int(match.group("ts"))
    tz = match.group("tz")
    sign = 1 if tz[0] == "+" else -1
    offset = timedelta(hours=int(tz[1:3]), minutes=int(tz[3:5])) * sign
    return datetime.fromtimestamp(ts, timezone(offset)).isoformat()


__all__ = ["LocalGitRepo", "CommitInfo", "validate_ref", "algorithm_for_hex", "oid_from_hex"]
