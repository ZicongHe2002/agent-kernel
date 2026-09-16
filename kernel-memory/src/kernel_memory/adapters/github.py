"""Read-only GitHub REST client with an injectable transport (specification section 8.1).

Public API
----------
``Response(status, headers, body)`` with ``header(name)`` (case-insensitive) and ``json()``
    (strict JSON, duplicate keys / NaN rejected).
``Transport`` protocol: ``request(method, url, headers) -> Response``.
``UrllibTransport(timeout=20.0)``
    Live HTTPS transport. Only ``GET``/``HEAD`` are allowed (anything else raises
    ``SecurityPolicyError``), redirects to another host are refused, network failures raise
    ``ExecutionInfrastructureError`` (``NETWORK_FAILURE``). Marked ``is_live = True``.
``FixtureTransport(routes)``
    Offline transport for tests: ``routes`` maps an exact URL (or the URL's path plus query,
    without scheme/host) to a list of ``Response`` objects that are served sequentially
    (the last one is repeated once the list is down to a single entry, so "403 then 200"
    and repeated collections both work). Every request is appended to ``.requests`` as
    ``{"method", "url", "headers"}`` for assertions. Unknown URLs yield a 404.
``fixture_response(status, payload, headers=None) -> Response`` JSON helper for fixtures.
``GitHubClient(transport, *, token=None, base_url="https://api.github.com", allow_network=False,
               sleeper=time.sleep, max_retries=4, per_page=100, max_pages=30, max_wait_seconds=60.0)``
    * A live transport requires ``allow_network=True``; otherwise the constructor raises
      ``PrerequisiteMissingError`` with code ``NETWORK_NOT_AUTHORIZED`` before any request.
    * Headers: ``Accept: application/vnd.github+json``, ``X-GitHub-Api-Version: 2022-11-28``,
      ``User-Agent: kernel-memory/0.2.0`` and ``Authorization: Bearer <token>`` when a token
      is given. The token is never logged or included in error details.
    * Status handling: 401 -> ``AuthorizationError`` (exit 7); 403/429 that carry
      ``X-RateLimit-Remaining: 0`` or ``Retry-After`` (or a rate-limit message) wait via
      ``sleeper`` (bounded by ``max_wait_seconds``) and retry up to ``max_retries``; other
      403 -> ``AuthorizationError``; 404 -> ``MissingReferenceError`` (``GITHUB_NOT_FOUND``);
      5xx -> exponential backoff retries then ``ExecutionInfrastructureError``; other 4xx ->
      ``InputError`` (``GITHUB_CLIENT_ERROR``).
    * Pagination follows ``Link: <...>; rel="next"`` with ``per_page`` and a hard ``max_pages``
      cap; hitting the cap is reported as partial coverage, never as complete.
    * ``get_repository(owner, repo) -> RepositoryInfo`` (``repo_uid`` = ``github:<host>:repo:<id>``)
    * ``get_pull(owner, repo, number) -> PullInfo`` (``github_state`` -> open/closed/merged)
    * ``list_pull_commits(owner, repo, number, pull=None) -> CommitEnumeration``. GitHub's
      pull-request commits endpoint returns at most 250 commits regardless of pagination
      (documented limit); coverage is ``partial`` when the PR reports more than 250 commits,
      when 250 items came back and more cannot be fetched, when pagination stopped at the
      page cap, or when the enumerated count disagrees with the PR's ``commits`` count.
    * ``compare_commits(owner, repo, base, head) -> CommitEnumeration`` via
      ``GET /repos/{o}/{r}/compare/{base}...{head}`` (also capped at 250 commits by GitHub;
      ``total_commits`` in the payload detects truncation).
    * No write method exists. Any non-GET/HEAD method reaching ``_request`` raises
      ``SecurityPolicyError``.
``github_state_from_pull(pull) -> "open" | "closed" | "merged" | "unknown"``.

Everything returned from the API (titles, bodies, messages) is data and is never
interpreted as instructions.
"""
from __future__ import annotations

import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from ..domain.errors import (
    AuthorizationError,
    ExecutionInfrastructureError,
    InputError,
    MissingReferenceError,
    PrerequisiteMissingError,
    SecurityPolicyError,
)
from ..domain.jsonio import dumps_compact, loads_strict

API_VERSION = "2022-11-28"
USER_AGENT = "kernel-memory/0.2.0"
ACCEPT = "application/vnd.github+json"
PR_COMMITS_LIMIT = 250  # GitHub returns at most 250 commits for a pull request (documented limit)
COMPARE_COMMITS_LIMIT = 250
READ_METHODS = frozenset({"GET", "HEAD"})
_NAME_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,99})$")
_LINK_RE = re.compile(r'<([^>]+)>\s*;\s*rel="([^"]+)"')
_MAX_BODY_BYTES = 20_000_000


# --------------------------------------------------------------------------------------
# Transport layer
# --------------------------------------------------------------------------------------
@dataclass
class Response:
    status: int
    headers: dict[str, str]
    body: bytes

    def header(self, name: str, default: str | None = None) -> str | None:
        wanted = name.lower()
        for key, value in self.headers.items():
            if key.lower() == wanted:
                return value
        return default

    def json(self) -> Any:
        if not self.body:
            return None
        return loads_strict(self.body)


class Transport(Protocol):
    def request(self, method: str, url: str, headers: dict[str, str]) -> Response: ...


def _check_method(method: str) -> str:
    upper = str(method).upper()
    if upper not in READ_METHODS:
        raise SecurityPolicyError(
            f"the GitHub client is read-only; HTTP method {method!r} is not permitted",
            code="GITHUB_WRITE_FORBIDDEN",
            details={"method": method},
        )
    return upper


class _SameHostRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse redirects that leave the original host or downgrade to plain HTTP."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        original = urllib.parse.urlsplit(req.full_url)
        target = urllib.parse.urlsplit(newurl)
        if target.scheme != "https" or (target.hostname or "").lower() != (original.hostname or "").lower():
            raise ExecutionInfrastructureError(
                "refusing to follow a redirect to a different host or insecure scheme",
                code="REDIRECT_REFUSED",
                details={"from_host": original.hostname, "to_host": target.hostname, "scheme": target.scheme},
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class UrllibTransport:
    is_live = True

    def __init__(self, timeout: float = 20.0) -> None:
        self.timeout = float(timeout)
        self._opener = urllib.request.build_opener(_SameHostRedirectHandler())

    def request(self, method: str, url: str, headers: dict[str, str]) -> Response:
        method = _check_method(method)
        parts = urllib.parse.urlsplit(url)
        if parts.scheme != "https":
            raise SecurityPolicyError(f"only https URLs are permitted, got {parts.scheme!r}", code="INSECURE_URL")
        req = urllib.request.Request(url, method=method, headers=dict(headers))
        try:
            with self._opener.open(req, timeout=self.timeout) as resp:  # noqa: S310 - https enforced above
                body = resp.read(_MAX_BODY_BYTES + 1)
                if len(body) > _MAX_BODY_BYTES:
                    raise InputError("GitHub response body exceeds the permitted size", code="RESPONSE_TOO_LARGE")
                return Response(status=resp.status, headers={k: v for k, v in resp.headers.items()}, body=body)
        except urllib.error.HTTPError as exc:
            body = exc.read() if exc.fp is not None else b""
            return Response(status=exc.code, headers={k: v for k, v in exc.headers.items()}, body=body)
        except ExecutionInfrastructureError:
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ExecutionInfrastructureError(
                f"network request to {parts.hostname} failed: {exc}", code="NETWORK_FAILURE", details={"host": parts.hostname}
            ) from exc


def fixture_response(status: int, payload: Any, headers: dict[str, str] | None = None) -> Response:
    """Build a JSON response for offline fixtures."""
    hdrs = {"Content-Type": "application/json; charset=utf-8"}
    if headers:
        hdrs.update(headers)
    body = dumps_compact(payload).encode("utf-8") if payload is not None else b""
    return Response(status=status, headers=hdrs, body=body)


def _route_keys(url: str) -> list[str]:
    parts = urllib.parse.urlsplit(url)
    path_query = parts.path + (f"?{parts.query}" if parts.query else "")
    keys = [url, path_query]
    if parts.query:
        keys.append(parts.path)
    return keys


class FixtureTransport:
    is_live = False

    def __init__(self, routes: dict[str, list[Response]] | Callable[[str, str], Response | None]) -> None:
        if callable(routes):
            self._resolver: Callable[[str, str], Response | None] | None = routes
            self._routes: dict[str, list[Response]] = {}
        else:
            self._resolver = None
            self._routes = {key: list(values) for key, values in routes.items()}
        self.requests: list[dict[str, Any]] = []

    def add(self, url: str, *responses: Response) -> None:
        self._routes.setdefault(url, []).extend(responses)

    def request(self, method: str, url: str, headers: dict[str, str]) -> Response:
        method = _check_method(method)
        self.requests.append({"method": method, "url": url, "headers": dict(headers)})
        if self._resolver is not None:
            resolved = self._resolver(method, url)
            if resolved is not None:
                return resolved
        for key in _route_keys(url):
            queue = self._routes.get(key)
            if queue:
                if len(queue) > 1:
                    return queue.pop(0)
                return queue[0]
        return fixture_response(404, {"message": "Not Found (fixture transport has no route)", "url": url})


# --------------------------------------------------------------------------------------
# Data returned to callers
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RepositoryInfo:
    id: int
    node_id: str
    full_name: str
    default_branch: str
    host: str

    @property
    def repo_uid(self) -> str:
        return f"github:{self.host}:repo:{self.id}"


@dataclass(frozen=True)
class PullInfo:
    number: int
    state: str
    merged: bool
    title: str
    body: str | None
    head_sha: str
    head_ref: str
    head_repo_full_name: str | None
    base_sha: str
    base_ref: str
    merge_commit_sha: str | None
    commits_count: int
    url: str
    updated_at: str | None

    @property
    def github_state(self) -> str:
        return github_state_from_pull(self)


@dataclass(frozen=True)
class ApiCommit:
    sha: str
    parents: list[str]
    message: str
    author_date: str | None
    tree_sha: str | None


@dataclass
class CommitEnumeration:
    commits: list[ApiCommit]
    coverage: str  # complete_for_snapshot | partial
    reason: str | None
    endpoint: str
    pages_fetched: int
    notes: list[str] = field(default_factory=list)

    @property
    def shas(self) -> list[str]:
        return [c.sha for c in self.commits]


def github_state_from_pull(pull: Any) -> str:
    merged = bool(getattr(pull, "merged", False))
    state = getattr(pull, "state", None)
    if merged:
        return "merged"
    if state in ("open", "closed"):
        return state
    return "unknown"


# --------------------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------------------
def _validate_name(value: str, what: str) -> str:
    if not isinstance(value, str) or not _NAME_RE.match(value) or ".." in value:
        raise InputError(f"invalid GitHub {what}: {value!r}", code="INVALID_GITHUB_NAME")
    return value


def _validate_number(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InputError(f"pull request number must be an integer >= 1, got {value!r}", code="INVALID_PR_NUMBER")
    return value


def _host_for(base_url: str) -> str:
    parts = urllib.parse.urlsplit(base_url)
    host = (parts.hostname or "").lower()
    if not host:
        raise InputError(f"base_url has no host: {base_url!r}", code="INVALID_BASE_URL")
    if host == "api.github.com":
        return "github.com"
    return host


class GitHubClient:
    def __init__(
        self,
        transport: Transport,
        *,
        token: str | None = None,
        base_url: str = "https://api.github.com",
        allow_network: bool = False,
        sleeper: Callable[[float], Any] = time.sleep,
        max_retries: int = 4,
        per_page: int = 100,
        max_pages: int = 30,
        max_wait_seconds: float = 60.0,
    ) -> None:
        if getattr(transport, "is_live", False) and not allow_network:
            raise PrerequisiteMissingError(
                "live GitHub requests require allow_network=true (permissions.allow_network); no request was made",
                code="NETWORK_NOT_AUTHORIZED",
                details={"transport": type(transport).__name__},
            )
        parts = urllib.parse.urlsplit(base_url)
        if parts.scheme != "https":
            raise SecurityPolicyError(f"GitHub base_url must use https, got {base_url!r}", code="INSECURE_URL")
        self._transport = transport
        self._token = token or None
        self.base_url = base_url.rstrip("/")
        self.host = _host_for(base_url)
        self.allow_network = bool(allow_network)
        self._sleeper = sleeper
        self.max_retries = max(0, int(max_retries))
        self.per_page = min(max(int(per_page), 1), 100)
        self.max_pages = max(1, int(max_pages))
        self.max_wait_seconds = max(0.0, float(max_wait_seconds))

    # ------------------------------------------------------------------ plumbing
    def _headers(self) -> dict[str, str]:
        headers = {"Accept": ACCEPT, "X-GitHub-Api-Version": API_VERSION, "User-Agent": USER_AGENT}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _url(self, path: str, query: dict[str, Any] | None = None) -> str:
        url = f"{self.base_url}{path}"
        if query:
            url += "?" + urllib.parse.urlencode(query)
        return url

    def _backoff(self, attempt: int) -> float:
        return min(0.5 * (2**attempt), self.max_wait_seconds)

    def _rate_limit_wait(self, response: Response, attempt: int) -> float | None:
        """Seconds to wait when the response is a rate limit, else None."""
        retry_after = response.header("Retry-After")
        remaining = response.header("X-RateLimit-Remaining")
        message = ""
        if response.body:
            try:
                payload = response.json()
                message = str(payload.get("message", "")) if isinstance(payload, dict) else ""
            except Exception:
                message = ""
        limited = retry_after is not None or remaining == "0" or "rate limit" in message.lower()
        if not limited:
            return None
        wait: float | None = None
        if retry_after is not None:
            try:
                wait = float(retry_after)
            except ValueError:
                wait = None
        if wait is None:
            reset = response.header("X-RateLimit-Reset")
            if reset is not None:
                try:
                    wait = float(reset) - time.time()
                except ValueError:
                    wait = None
        if wait is None:
            wait = self._backoff(attempt)
        return min(max(wait, 0.0), self.max_wait_seconds)

    def _request(self, method: str, url: str) -> Response:
        method = _check_method(method)
        if getattr(self._transport, "is_live", False) and not self.allow_network:
            raise PrerequisiteMissingError("live GitHub requests are not authorized", code="NETWORK_NOT_AUTHORIZED")
        attempt = 0
        while True:
            response = self._transport.request(method, url, self._headers())
            status = response.status
            if 200 <= status < 300:
                return response
            if status == 401:
                raise AuthorizationError(
                    "GitHub rejected the credentials (401); check the configured token", code="GITHUB_UNAUTHORIZED", details={"url": url}
                )
            if status == 404:
                raise MissingReferenceError(f"GitHub resource not found: {url}", code="GITHUB_NOT_FOUND", details={"url": url})
            if status in (403, 429):
                wait = self._rate_limit_wait(response, attempt)
                if wait is None:
                    raise AuthorizationError(
                        f"GitHub denied access ({status}) to {url}", code="GITHUB_FORBIDDEN", details={"url": url, "status": status}
                    )
                if attempt >= self.max_retries:
                    raise ExecutionInfrastructureError(
                        f"GitHub rate limit persisted after {attempt} retries for {url}",
                        code="GITHUB_RATE_LIMITED",
                        details={"url": url, "status": status},
                    )
                self._sleeper(wait)
                attempt += 1
                continue
            if 500 <= status < 600:
                if attempt >= self.max_retries:
                    raise ExecutionInfrastructureError(
                        f"GitHub server error {status} persisted after {attempt} retries for {url}",
                        code="GITHUB_SERVER_ERROR",
                        details={"url": url, "status": status},
                    )
                self._sleeper(self._backoff(attempt))
                attempt += 1
                continue
            raise InputError(
                f"GitHub returned status {status} for {url}",
                code="GITHUB_CLIENT_ERROR",
                details={"url": url, "status": status, "body": response.body[:500].decode("utf-8", errors="replace")},
            )

    def _get_json(self, url: str) -> tuple[Any, Response]:
        response = self._request("GET", url)
        try:
            payload = response.json()
        except InputError as exc:
            raise ExecutionInfrastructureError(f"GitHub returned invalid JSON for {url}: {exc}", code="GITHUB_INVALID_JSON") from exc
        return payload, response

    def _paginate(self, url: str) -> tuple[list[Any], int, bool, list[dict[str, Any]]]:
        """Follow rel=next links. Returns (items, pages_fetched, truncated_at_cap, raw_pages)."""
        items: list[Any] = []
        raw_pages: list[dict[str, Any]] = []
        pages = 0
        next_url: str | None = url
        truncated = False
        while next_url is not None:
            if pages >= self.max_pages:
                truncated = True
                break
            payload, response = self._get_json(next_url)
            pages += 1
            if isinstance(payload, list):
                items.extend(payload)
            elif isinstance(payload, dict):
                raw_pages.append(payload)
                commits = payload.get("commits")
                if isinstance(commits, list):
                    items.extend(commits)
            else:
                raise ExecutionInfrastructureError(f"unexpected GitHub payload type for {next_url}", code="GITHUB_INVALID_JSON")
            next_url = _next_link(response.header("Link"), next_url)
        return items, pages, truncated, raw_pages

    # ------------------------------------------------------------------ public read methods
    def get_repository(self, owner: str, repo: str) -> RepositoryInfo:
        owner, repo = _validate_name(owner, "owner"), _validate_name(repo, "repository")
        payload, _ = self._get_json(self._url(f"/repos/{owner}/{repo}"))
        if not isinstance(payload, dict):
            raise ExecutionInfrastructureError("repository payload is not an object", code="GITHUB_INVALID_JSON")
        repo_id = payload.get("id")
        if isinstance(repo_id, bool) or not isinstance(repo_id, int):
            raise ExecutionInfrastructureError("repository payload lacks a numeric id", code="GITHUB_INVALID_JSON")
        return RepositoryInfo(
            id=repo_id,
            node_id=str(payload.get("node_id", "")),
            full_name=str(payload.get("full_name", f"{owner}/{repo}")),
            default_branch=str(payload.get("default_branch", "")),
            host=self.host,
        )

    def get_pull(self, owner: str, repo: str, number: int) -> PullInfo:
        owner, repo = _validate_name(owner, "owner"), _validate_name(repo, "repository")
        number = _validate_number(number)
        payload, _ = self._get_json(self._url(f"/repos/{owner}/{repo}/pulls/{number}"))
        if not isinstance(payload, dict):
            raise ExecutionInfrastructureError("pull request payload is not an object", code="GITHUB_INVALID_JSON")
        head = payload.get("head") or {}
        base = payload.get("base") or {}
        head_repo = head.get("repo") if isinstance(head, dict) else None
        head_sha = _require_sha(head.get("sha"), "head.sha")
        base_sha = _require_sha(base.get("sha"), "base.sha")
        merge_sha = payload.get("merge_commit_sha")
        commits_count = payload.get("commits", 0)
        if isinstance(commits_count, bool) or not isinstance(commits_count, int):
            commits_count = 0
        return PullInfo(
            number=int(payload.get("number", number)),
            state=str(payload.get("state", "unknown")),
            merged=bool(payload.get("merged", False)),
            title=str(payload.get("title", "")),
            body=payload.get("body") if isinstance(payload.get("body"), str) else None,
            head_sha=head_sha,
            head_ref=str(head.get("ref", "")),
            head_repo_full_name=str(head_repo["full_name"]) if isinstance(head_repo, dict) and head_repo.get("full_name") else None,
            base_sha=base_sha,
            base_ref=str(base.get("ref", "")),
            merge_commit_sha=_require_sha(merge_sha, "merge_commit_sha") if isinstance(merge_sha, str) and merge_sha else None,
            commits_count=commits_count,
            url=str(payload.get("html_url") or payload.get("url") or ""),
            updated_at=str(payload["updated_at"]) if payload.get("updated_at") else None,
        )

    def list_pull_commits(self, owner: str, repo: str, number: int, pull: PullInfo | None = None) -> CommitEnumeration:
        owner, repo = _validate_name(owner, "owner"), _validate_name(repo, "repository")
        number = _validate_number(number)
        endpoint = f"/repos/{owner}/{repo}/pulls/{number}/commits"
        items, pages, truncated, _ = self._paginate(self._url(endpoint, {"per_page": self.per_page}))
        commits = [_api_commit(item) for item in items]
        notes: list[str] = []
        coverage, reason = "complete_for_snapshot", None
        if pull is not None and pull.commits_count > PR_COMMITS_LIMIT:
            coverage = "partial"
            reason = (
                f"GitHub's pull request commits endpoint returns at most {PR_COMMITS_LIMIT} commits; "
                f"the pull request reports {pull.commits_count} commits ({len(commits)} enumerated)"
            )
        elif len(commits) >= PR_COMMITS_LIMIT:
            coverage = "partial"
            reason = (
                f"{len(commits)} commits were returned, which is GitHub's {PR_COMMITS_LIMIT}-commit limit for the "
                "pull request commits endpoint; further pagination cannot return more"
            )
        elif truncated:
            coverage = "partial"
            reason = f"pagination stopped at the page cap ({self.max_pages} pages) with a next link still present"
        elif pull is not None and pull.commits_count != len(commits):
            coverage = "partial"
            reason = f"enumerated {len(commits)} commits but the pull request reports {pull.commits_count}"
        if pull is not None and pull.head_sha and commits and commits[-1].sha != pull.head_sha:
            notes.append("last enumerated commit differs from the pull request head; the API order is not guaranteed")
        return CommitEnumeration(commits=commits, coverage=coverage, reason=reason, endpoint=endpoint, pages_fetched=pages, notes=notes)

    def compare_commits(self, owner: str, repo: str, base: str, head: str) -> CommitEnumeration:
        owner, repo = _validate_name(owner, "owner"), _validate_name(repo, "repository")
        base_q = urllib.parse.quote(_validate_revision(base, "base"), safe="")
        head_q = urllib.parse.quote(_validate_revision(head, "head"), safe="")
        endpoint = f"/repos/{owner}/{repo}/compare/{base_q}...{head_q}"
        items, pages, truncated, raw_pages = self._paginate(self._url(endpoint, {"per_page": self.per_page}))
        commits = [_api_commit(item) for item in items]
        total = None
        if raw_pages:
            candidate = raw_pages[0].get("total_commits")
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                total = candidate
        coverage, reason = "complete_for_snapshot", None
        if total is not None and total > len(commits):
            coverage = "partial"
            reason = (
                f"compare endpoint reports total_commits={total} but returned {len(commits)} "
                f"(GitHub caps the compare commit list at {COMPARE_COMMITS_LIMIT})"
            )
        elif len(commits) >= COMPARE_COMMITS_LIMIT and total is None:
            coverage = "partial"
            reason = f"{len(commits)} commits returned, which is GitHub's {COMPARE_COMMITS_LIMIT}-commit compare limit"
        elif truncated:
            coverage = "partial"
            reason = f"pagination stopped at the page cap ({self.max_pages} pages) with a next link still present"
        elif total is None:
            coverage = "partial"
            reason = "compare payload lacks total_commits; completeness cannot be verified"
        return CommitEnumeration(commits=commits, coverage=coverage, reason=reason, endpoint=endpoint, pages_fetched=pages)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _next_link(link_header: str | None, current_url: str) -> str | None:
    if not link_header:
        return None
    current_host = urllib.parse.urlsplit(current_url).hostname
    for url, rel in _LINK_RE.findall(link_header):
        if rel.strip().lower() == "next":
            if urllib.parse.urlsplit(url).hostname != current_host:
                raise SecurityPolicyError("pagination link points to a different host; refusing to follow", code="LINK_HOST_MISMATCH")
            return url
    return None


def _require_sha(value: Any, what: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value):
        raise ExecutionInfrastructureError(f"GitHub payload field {what} is not a full object id: {value!r}", code="GITHUB_INVALID_JSON")
    return value


def _validate_revision(value: str, what: str) -> str:
    if not isinstance(value, str) or not value or value.startswith("-") or ".." in value or any(c.isspace() for c in value):
        raise InputError(f"invalid {what} revision for compare: {value!r}", code="INVALID_REVISION")
    return value


def _api_commit(item: Any) -> ApiCommit:
    if not isinstance(item, dict):
        raise ExecutionInfrastructureError("commit list entry is not an object", code="GITHUB_INVALID_JSON")
    sha = _require_sha(item.get("sha"), "sha")
    parents_raw = item.get("parents") or []
    parents = [_require_sha(p.get("sha") if isinstance(p, dict) else p, "parents[].sha") for p in parents_raw]
    commit = item.get("commit") or {}
    message = str(commit.get("message", "")) if isinstance(commit, dict) else ""
    author = commit.get("author") if isinstance(commit, dict) else None
    author_date = str(author["date"]) if isinstance(author, dict) and author.get("date") else None
    tree = commit.get("tree") if isinstance(commit, dict) else None
    tree_sha = tree.get("sha") if isinstance(tree, dict) and isinstance(tree.get("sha"), str) else None
    return ApiCommit(sha=sha, parents=parents, message=message, author_date=author_date, tree_sha=tree_sha)


__all__ = [
    "API_VERSION",
    "USER_AGENT",
    "PR_COMMITS_LIMIT",
    "COMPARE_COMMITS_LIMIT",
    "Response",
    "Transport",
    "UrllibTransport",
    "FixtureTransport",
    "fixture_response",
    "RepositoryInfo",
    "PullInfo",
    "ApiCommit",
    "CommitEnumeration",
    "GitHubClient",
    "github_state_from_pull",
]
