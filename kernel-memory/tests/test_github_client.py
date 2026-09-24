"""Read-only GitHub client over offline HTTP fixtures (specification section 8.1; T09).

Every response comes from ``tests/fixtures/github/*.json`` through ``FixtureTransport``; no test
opens a network connection. Pagination, rate-limit backoff, retries, the read-only guarantee,
token handling, and the coverage verdicts of the commit enumerations are covered here.
"""
from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from kernel_memory.adapters import github as github_module
from kernel_memory.adapters.github import (
    COMPARE_COMMITS_LIMIT,
    PR_COMMITS_LIMIT,
    FixtureTransport,
    GitHubClient,
    PullInfo,
    Response,
    UrllibTransport,
    fixture_response,
    github_state_from_pull,
)
from kernel_memory.domain.errors import (
    AuthorizationError,
    ExecutionInfrastructureError,
    InputError,
    MissingReferenceError,
    PrerequisiteMissingError,
    SecurityPolicyError,
)
from kernel_memory.domain.jsonio import load_json_file

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "github"
API = "https://api.github.com"
OWNER, REPO = "acme", "kernels"
REPO_PATH = f"/repos/{OWNER}/{REPO}"
PULL_101_PATH = f"{REPO_PATH}/pulls/101"
COMMITS_101_PATH = f"{PULL_101_PATH}/commits"

BASE_SHA = "ba5e" * 10
C1, C2, C3 = "c1" * 20, "c2" * 20, "c3" * 20
MERGE_101 = "3e" * 20
TOKEN = "ghp_fixtureSecretToken0123456789"


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def canned_envelope(name: str) -> dict[str, Any]:
    return load_json_file(FIXTURE_DIR / f"{name}.json")


def canned(name: str) -> Response:
    data = canned_envelope(name)
    return fixture_response(int(data["status"]), data.get("body"), dict(data.get("headers") or {}))


def canned_body(name: str) -> Any:
    return json.loads(json.dumps(canned_envelope(name)["body"]))


class SleepRecorder:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(float(seconds))


def pr101_routes() -> dict[str, list[Response]]:
    return {
        REPO_PATH: [canned("repository_900001")],
        PULL_101_PATH: [canned("pull_101")],
        f"{COMMITS_101_PATH}?per_page=1": [canned("pull_101_commits_page_1")],
        f"{COMMITS_101_PATH}?per_page=1&page=2": [canned("pull_101_commits_page_2")],
        f"{COMMITS_101_PATH}?per_page=1&page=3": [canned("pull_101_commits_page_3")],
    }


def make_client(routes: dict[str, list[Response]], **kwargs: Any) -> tuple[GitHubClient, FixtureTransport, SleepRecorder]:
    transport = FixtureTransport(routes)
    sleeper = SleepRecorder()
    kwargs.setdefault("per_page", 1)
    client = GitHubClient(transport, sleeper=sleeper, **kwargs)
    return client, transport, sleeper


def refuse_network(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Make any attempt to open a real connection fail the test, and record attempts."""
    attempts: list[Any] = []

    def _open(self: Any, req: Any, *args: Any, **kwargs: Any) -> Any:
        attempts.append(req)
        raise AssertionError("a real network connection was attempted")

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", _open)
    return attempts


# --------------------------------------------------------------------------------------
# transport primitives
# --------------------------------------------------------------------------------------
def test_response_header_lookup_is_case_insensitive_and_json_is_strict() -> None:
    response = fixture_response(200, {"a": 1}, {"X-RateLimit-Remaining": "0"})
    assert response.header("x-ratelimit-remaining") == "0"
    assert response.header("Missing", "dflt") == "dflt"
    assert response.json() == {"a": 1}
    assert fixture_response(204, None).json() is None
    with pytest.raises(InputError):
        Response(200, {}, b'{"a": 1, "a": 2}').json()


def test_fixture_transport_serves_sequentially_records_requests_and_404s_unknown_urls() -> None:
    transport = FixtureTransport({REPO_PATH: [fixture_response(500, {"m": 1}), fixture_response(200, {"m": 2})]})
    first = transport.request("GET", f"{API}{REPO_PATH}", {"A": "b"})
    second = transport.request("get", f"{API}{REPO_PATH}", {})
    third = transport.request("GET", f"{API}{REPO_PATH}", {})
    assert (first.status, second.status, third.status) == (500, 200, 200)
    assert transport.request("GET", f"{API}/nowhere", {}).status == 404
    assert [r["method"] for r in transport.requests] == ["GET", "GET", "GET", "GET"]
    assert transport.requests[0]["headers"] == {"A": "b"} and transport.requests[0]["url"] == f"{API}{REPO_PATH}"
    with pytest.raises(SecurityPolicyError) as info:
        transport.request("PUT", f"{API}{REPO_PATH}", {})
    assert info.value.code == "GITHUB_WRITE_FORBIDDEN" and info.value.exit_code == 7


# --------------------------------------------------------------------------------------
# pagination
# --------------------------------------------------------------------------------------
def test_pagination_follows_link_header_across_three_pages() -> None:
    client, transport, sleeper = make_client(pr101_routes())
    pull = client.get_pull(OWNER, REPO, 101)
    enumeration = client.list_pull_commits(OWNER, REPO, 101, pull=pull)

    assert enumeration.shas == [C1, C2, C3]
    assert enumeration.pages_fetched == 3
    assert enumeration.coverage == "complete_for_snapshot" and enumeration.reason is None
    assert enumeration.endpoint == COMMITS_101_PATH
    assert enumeration.notes == []
    assert [c.parents for c in enumeration.commits] == [[BASE_SHA], [C1], [C2]]
    assert enumeration.commits[0].message.startswith("Introduce tiled loop skeleton")
    assert enumeration.commits[0].author_date == "2026-09-09T10:00:00Z"
    urls = [r["url"] for r in transport.requests if "/commits" in r["url"]]
    assert urls == [
        f"{API}{COMMITS_101_PATH}?per_page=1",
        f"{API}{COMMITS_101_PATH}?per_page=1&page=2",
        f"{API}{COMMITS_101_PATH}?per_page=1&page=3",
    ]
    for request in transport.requests:
        assert request["headers"]["Accept"] == "application/vnd.github+json"
        assert request["headers"]["X-GitHub-Api-Version"] == "2022-11-28"
        assert request["headers"]["User-Agent"].startswith("kernel-memory/")
    assert sleeper.calls == []


def test_t09_pagination_stopping_at_max_pages_is_partial() -> None:
    client, transport, _ = make_client(pr101_routes(), max_pages=2)
    pull = client.get_pull(OWNER, REPO, 101)
    enumeration = client.list_pull_commits(OWNER, REPO, 101, pull=pull)
    assert enumeration.coverage == "partial"
    assert enumeration.pages_fetched == 2 and enumeration.shas == [C1, C2]
    assert enumeration.reason is not None and "page cap" in enumeration.reason and "2 pages" in enumeration.reason
    assert sum(1 for r in transport.requests if "/commits" in r["url"]) == 2


def test_pagination_link_to_another_host_is_refused() -> None:
    routes = pr101_routes()
    routes[f"{COMMITS_101_PATH}?per_page=1"] = [
        fixture_response(200, canned_body("pull_101_commits_page_1"), {"Link": '<https://evil.example/repos/x/commits?page=2>; rel="next"'})
    ]
    client, transport, _ = make_client(routes)
    with pytest.raises(SecurityPolicyError) as info:
        client.list_pull_commits(OWNER, REPO, 101)
    assert info.value.code == "LINK_HOST_MISMATCH" and info.value.exit_code == 7
    assert all("evil.example" not in r["url"] for r in transport.requests)


# --------------------------------------------------------------------------------------
# rate limits, retries, status mapping
# --------------------------------------------------------------------------------------
def test_403_rate_limit_with_retry_after_then_200_sleeps_bounded_and_succeeds() -> None:
    routes = {REPO_PATH: [canned("rate_limited_403_retry_after"), canned("repository_900001")]}
    client, transport, sleeper = make_client(routes, max_wait_seconds=60.0)
    repository = client.get_repository(OWNER, REPO)
    assert repository.id == 900001
    assert sleeper.calls == [7.0]
    assert len(transport.requests) == 2


def test_429_rate_limit_with_reset_far_in_future_is_bounded_by_max_wait() -> None:
    routes = {REPO_PATH: [canned("rate_limited_429_reset"), canned("repository_900001")]}
    client, transport, sleeper = make_client(routes, max_wait_seconds=30.0)
    assert client.get_repository(OWNER, REPO).id == 900001
    assert sleeper.calls == [30.0]
    assert len(transport.requests) == 2


def test_persistent_rate_limit_stops_after_max_retries_with_infrastructure_error() -> None:
    routes = {REPO_PATH: [canned("rate_limited_403_retry_after")]}
    client, transport, sleeper = make_client(routes, max_retries=2)
    with pytest.raises(ExecutionInfrastructureError) as info:
        client.get_repository(OWNER, REPO)
    assert info.value.code == "GITHUB_RATE_LIMITED" and info.value.exit_code == 6
    assert sleeper.calls == [7.0, 7.0]
    assert len(transport.requests) == 3


def test_plain_403_is_authorization_error_without_sleeping() -> None:
    client, transport, sleeper = make_client({REPO_PATH: [canned("forbidden_403")]})
    with pytest.raises(AuthorizationError) as info:
        client.get_repository(OWNER, REPO)
    assert info.value.code == "GITHUB_FORBIDDEN" and info.value.exit_code == 7
    assert sleeper.calls == [] and len(transport.requests) == 1


def test_500_then_200_is_retried_with_backoff() -> None:
    routes = {REPO_PATH: [canned("server_error_500"), canned("repository_900001")]}
    client, transport, sleeper = make_client(routes)
    assert client.get_repository(OWNER, REPO).full_name == "acme/kernels"
    assert sleeper.calls == [0.5]
    assert len(transport.requests) == 2


def test_persistent_500_raises_infrastructure_error_after_max_retries() -> None:
    client, transport, sleeper = make_client({REPO_PATH: [canned("server_error_500")]}, max_retries=2)
    with pytest.raises(ExecutionInfrastructureError) as info:
        client.get_repository(OWNER, REPO)
    assert info.value.code == "GITHUB_SERVER_ERROR" and info.value.exit_code == 6
    assert info.value.details["status"] == 500
    assert sleeper.calls == [0.5, 1.0]
    assert len(transport.requests) == 3


def test_404_is_missing_reference_exit_3() -> None:
    client, _, sleeper = make_client({REPO_PATH: [canned("not_found_404")]})
    with pytest.raises(MissingReferenceError) as info:
        client.get_repository(OWNER, REPO)
    assert info.value.code == "GITHUB_NOT_FOUND" and info.value.exit_code == 3
    assert sleeper.calls == []
    # An unrouted URL is a 404 from the fixture transport and maps the same way.
    client, _, _ = make_client({})
    with pytest.raises(MissingReferenceError) as info:
        client.get_pull(OWNER, REPO, 999)
    assert info.value.code == "GITHUB_NOT_FOUND"


def test_401_is_authorization_error_exit_7() -> None:
    client, _, sleeper = make_client({REPO_PATH: [canned("unauthorized_401")]}, token=TOKEN)
    with pytest.raises(AuthorizationError) as info:
        client.get_repository(OWNER, REPO)
    assert info.value.code == "GITHUB_UNAUTHORIZED" and info.value.exit_code == 7
    assert sleeper.calls == []


def test_other_4xx_is_input_error_exit_2() -> None:
    client, _, _ = make_client({PULL_101_PATH: [canned("unprocessable_422")]})
    with pytest.raises(InputError) as info:
        client.get_pull(OWNER, REPO, 101)
    assert info.value.code == "GITHUB_CLIENT_ERROR" and info.value.exit_code == 2
    assert info.value.details["status"] == 422


# --------------------------------------------------------------------------------------
# network authorization and the read-only guarantee
# --------------------------------------------------------------------------------------
def test_live_transport_without_allow_network_is_refused_before_any_request(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = refuse_network(monkeypatch)
    called: list[Any] = []
    monkeypatch.setattr(UrllibTransport, "request", lambda self, *a, **k: called.append(a) or fixture_response(200, {}))
    with pytest.raises(PrerequisiteMissingError) as info:
        GitHubClient(UrllibTransport(), allow_network=False)
    assert info.value.code == "NETWORK_NOT_AUTHORIZED" and info.value.exit_code == 5
    assert "no request was made" in info.value.message
    assert called == [] and attempts == []


def test_allow_network_default_is_false_and_the_request_guard_also_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = refuse_network(monkeypatch)
    with pytest.raises(PrerequisiteMissingError):
        GitHubClient(UrllibTransport())  # default allow_network=False
    # A client that was authorized at construction but lost the permission later refuses too.
    client = GitHubClient(UrllibTransport(), allow_network=True)
    client.allow_network = False
    with pytest.raises(PrerequisiteMissingError) as info:
        client.get_repository(OWNER, REPO)
    assert info.value.code == "NETWORK_NOT_AUTHORIZED" and attempts == []


def test_urllib_transport_refuses_post_without_opening_a_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = refuse_network(monkeypatch)
    transport = UrllibTransport()
    with pytest.raises(SecurityPolicyError) as info:
        transport.request("POST", f"{API}{REPO_PATH}/issues", {"Accept": "application/vnd.github+json"})
    assert info.value.code == "GITHUB_WRITE_FORBIDDEN" and info.value.exit_code == 7
    assert info.value.details == {"method": "POST"}
    for method in ("PUT", "PATCH", "DELETE"):
        with pytest.raises(SecurityPolicyError):
            transport.request(method, f"{API}{REPO_PATH}", {})
    with pytest.raises(SecurityPolicyError) as info:
        transport.request("GET", f"http://api.github.com{REPO_PATH}", {})
    assert info.value.code == "INSECURE_URL"
    assert attempts == []


def test_client_never_issues_a_write_even_through_a_permissive_transport() -> None:
    client, transport, _ = make_client(pr101_routes())
    with pytest.raises(SecurityPolicyError) as info:
        client._request("DELETE", f"{API}{REPO_PATH}")
    assert info.value.code == "GITHUB_WRITE_FORBIDDEN" and info.value.exit_code == 7
    assert transport.requests == []
    assert not any(name.startswith(("create", "update", "delete", "post", "put", "patch", "merge")) for name in dir(GitHubClient))
    with pytest.raises(SecurityPolicyError) as info:
        GitHubClient(FixtureTransport({}), base_url="http://api.github.com")
    assert info.value.code == "INSECURE_URL"


def test_authorization_header_only_when_a_token_is_given() -> None:
    client, transport, _ = make_client(pr101_routes(), token=TOKEN)
    client.get_repository(OWNER, REPO)
    assert transport.requests[0]["headers"]["Authorization"] == f"Bearer {TOKEN}"

    client, transport, _ = make_client(pr101_routes())
    client.get_repository(OWNER, REPO)
    assert "Authorization" not in transport.requests[0]["headers"]
    assert "authorization" not in {k.lower() for k in transport.requests[0]["headers"]}

    client, transport, _ = make_client(pr101_routes(), token="")
    client.get_repository(OWNER, REPO)
    assert "Authorization" not in transport.requests[0]["headers"]


@pytest.mark.parametrize("fixture", ["unauthorized_401", "not_found_404", "forbidden_403", "server_error_500", "unprocessable_422"])
def test_token_never_appears_in_exceptions_or_logs(fixture: str, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    client, _, _ = make_client({REPO_PATH: [canned(fixture)]}, token=TOKEN, max_retries=1)
    with pytest.raises(Exception) as info:  # noqa: PT011 - every mapped error class is checked for leaks
        client.get_repository(OWNER, REPO)
    error = info.value
    assert TOKEN not in str(error) and TOKEN not in repr(error)
    assert TOKEN not in json.dumps(error.to_dict())
    assert TOKEN not in caplog.text
    assert not any(TOKEN in str(getattr(record, "args", "")) for record in caplog.records)


def test_token_not_in_fixture_transport_404_fallback_message() -> None:
    client, _, _ = make_client({}, token=TOKEN)
    with pytest.raises(MissingReferenceError) as info:
        client.get_repository(OWNER, REPO)
    assert TOKEN not in json.dumps(info.value.to_dict())


# --------------------------------------------------------------------------------------
# repository / pull mapping
# --------------------------------------------------------------------------------------
def test_get_repository_maps_repo_uid() -> None:
    client, _, _ = make_client(pr101_routes())
    repository = client.get_repository(OWNER, REPO)
    assert repository.repo_uid == "github:github.com:repo:900001"
    assert (repository.id, repository.full_name, repository.default_branch, repository.host) == (900001, "acme/kernels", "main", "github.com")
    assert repository.node_id == "R_kgDOAA24Cw"


def test_get_repository_on_enterprise_host_uses_that_host_in_repo_uid() -> None:
    base_url = "https://ghe.example.com/api/v3"
    transport = FixtureTransport({f"{base_url}{REPO_PATH}": [canned("repository_900001")]})
    client = GitHubClient(transport, base_url=base_url)
    assert client.get_repository(OWNER, REPO).repo_uid == "github:ghe.example.com:repo:900001"
    assert transport.requests[0]["url"] == f"{base_url}{REPO_PATH}"


def test_get_repository_rejects_a_payload_without_numeric_id() -> None:
    client, _, _ = make_client({REPO_PATH: [fixture_response(200, {"id": "900001", "full_name": "acme/kernels"})]})
    with pytest.raises(ExecutionInfrastructureError) as info:
        client.get_repository(OWNER, REPO)
    assert info.value.code == "GITHUB_INVALID_JSON" and info.value.exit_code == 6


@pytest.mark.parametrize("bad_owner,bad_repo", [("acme/evil", "kernels"), ("acme", "-kernels"), ("..", "kernels"), ("acme", "kern els"), ("", "kernels")])
def test_invalid_owner_or_repo_is_rejected_before_any_request(bad_owner: str, bad_repo: str) -> None:
    client, transport, _ = make_client(pr101_routes())
    with pytest.raises(InputError) as info:
        client.get_repository(bad_owner, bad_repo)
    assert info.value.code == "INVALID_GITHUB_NAME" and info.value.exit_code == 2
    assert transport.requests == []


@pytest.mark.parametrize("number", [0, -1, True, "101", 1.5])
def test_invalid_pull_number_is_rejected_before_any_request(number: Any) -> None:
    client, transport, _ = make_client(pr101_routes())
    with pytest.raises(InputError) as info:
        client.get_pull(OWNER, REPO, number)
    assert info.value.code == "INVALID_PR_NUMBER" and info.value.exit_code == 2
    assert transport.requests == []


@pytest.mark.parametrize(
    "state,merged,expected",
    [("open", False, "open"), ("closed", False, "closed"), ("closed", True, "merged"), ("open", True, "merged")],
)
def test_get_pull_maps_github_state_and_keeps_merge_sha_separate_from_head(state: str, merged: bool, expected: str) -> None:
    body = canned_body("pull_101")
    body["state"], body["merged"] = state, merged
    client, _, _ = make_client({PULL_101_PATH: [fixture_response(200, body)]})
    pull = client.get_pull(OWNER, REPO, 101)
    assert isinstance(pull, PullInfo)
    assert pull.github_state == expected == github_state_from_pull(pull)
    assert pull.number == 101 and pull.commits_count == 3
    assert pull.head_sha == C3 and pull.base_sha == BASE_SHA
    assert pull.merge_commit_sha == MERGE_101 and pull.merge_commit_sha != pull.head_sha
    assert pull.head_ref == "feature/tile-inner-loop" and pull.base_ref == "main"
    assert pull.head_repo_full_name == "acme/kernels"
    assert pull.title == body["title"] and pull.body == body["body"]
    assert pull.url == "https://github.com/acme/kernels/pull/101" and pull.updated_at == "2026-09-10T08:00:00Z"


def test_github_state_unknown_for_unrecognised_state_and_null_merge_sha_is_none() -> None:
    body = canned_body("pull_101")
    body["state"], body["merged"], body["merge_commit_sha"] = "draft?", False, None
    client, _, _ = make_client({PULL_101_PATH: [fixture_response(200, body)]})
    pull = client.get_pull(OWNER, REPO, 101)
    assert pull.github_state == "unknown" and pull.merge_commit_sha is None
    assert github_state_from_pull(object()) == "unknown"


@pytest.mark.parametrize("field", ["head", "base"])
def test_get_pull_rejects_short_or_missing_object_ids_never_pads(field: str) -> None:
    body = canned_body("pull_101")
    body[field]["sha"] = C3[:12]
    client, _, _ = make_client({PULL_101_PATH: [fixture_response(200, body)]})
    with pytest.raises(ExecutionInfrastructureError) as info:
        client.get_pull(OWNER, REPO, 101)
    assert info.value.code == "GITHUB_INVALID_JSON" and info.value.exit_code == 6


def test_get_pull_rejects_invalid_json_body() -> None:
    client, _, _ = make_client({PULL_101_PATH: [Response(200, {"Content-Type": "application/json"}, b"{not json")]})
    with pytest.raises(ExecutionInfrastructureError) as info:
        client.get_pull(OWNER, REPO, 101)
    assert info.value.code == "GITHUB_INVALID_JSON"


# --------------------------------------------------------------------------------------
# T09: coverage of commit enumerations
# --------------------------------------------------------------------------------------
def test_t09_pull_reporting_300_commits_is_partial_with_reason_mentioning_250() -> None:
    routes = {
        f"{REPO_PATH}/pulls/103": [canned("pull_103_large")],
        f"{REPO_PATH}/pulls/103/commits?per_page=100": [canned("pull_103_commits")],
    }
    client, _, _ = make_client(routes, per_page=100)
    pull = client.get_pull(OWNER, REPO, 103)
    assert pull.commits_count == 300
    enumeration = client.list_pull_commits(OWNER, REPO, 103, pull=pull)
    assert enumeration.coverage == "partial"
    assert enumeration.reason is not None
    assert str(PR_COMMITS_LIMIT) == "250" and "250" in enumeration.reason and "300" in enumeration.reason
    assert len(enumeration.commits) == 3 and enumeration.pages_fetched == 1
    assert enumeration.notes == []  # last enumerated commit is the head


def test_t09_exactly_250_returned_commits_is_partial_even_without_pull_info() -> None:
    base = canned_body("pull_101_commits_page_1")[0]
    items = []
    previous = BASE_SHA
    for index in range(PR_COMMITS_LIMIT):
        sha = f"{index:040x}"
        item = json.loads(json.dumps(base))
        item["sha"], item["parents"] = sha, [{"sha": previous}]
        items.append(item)
        previous = sha
    client, _, _ = make_client({f"{COMMITS_101_PATH}?per_page=100": [fixture_response(200, items)]}, per_page=100)
    enumeration = client.list_pull_commits(OWNER, REPO, 101)
    assert enumeration.coverage == "partial" and len(enumeration.commits) == 250
    assert enumeration.reason is not None and "250" in enumeration.reason


def test_t09_count_mismatch_with_pull_is_partial_and_head_order_note_is_recorded() -> None:
    body = canned_body("pull_101")
    body["commits"] = 4
    routes = pr101_routes()
    routes[PULL_101_PATH] = [fixture_response(200, body)]
    client, _, _ = make_client(routes)
    pull = client.get_pull(OWNER, REPO, 101)
    enumeration = client.list_pull_commits(OWNER, REPO, 101, pull=pull)
    assert enumeration.coverage == "partial"
    assert enumeration.reason == "enumerated 3 commits but the pull request reports 4"
    # Head not last in the enumeration -> a note, not a coverage change.
    body["commits"], body["head"]["sha"] = 3, C1
    routes[PULL_101_PATH] = [fixture_response(200, body)]
    client, _, _ = make_client(routes)
    enumeration = client.list_pull_commits(OWNER, REPO, 101, pull=client.get_pull(OWNER, REPO, 101))
    assert enumeration.coverage == "complete_for_snapshot"
    assert any("differs from the pull request head" in note for note in enumeration.notes)


def test_t09_compare_with_total_commits_greater_than_returned_is_partial() -> None:
    compare_path = f"{REPO_PATH}/compare/{BASE_SHA}...{C2}"
    client, transport, _ = make_client({compare_path: [canned("compare_partial")]}, per_page=100)
    enumeration = client.compare_commits(OWNER, REPO, BASE_SHA, C2)
    assert enumeration.coverage == "partial" and enumeration.shas == [C1, C2]
    assert enumeration.reason is not None and "total_commits=5" in enumeration.reason and str(COMPARE_COMMITS_LIMIT) in enumeration.reason
    assert enumeration.endpoint == compare_path and enumeration.pages_fetched == 1
    assert transport.requests[0]["url"] == f"{API}{compare_path}?per_page=100"


def test_compare_with_matching_total_is_complete_and_missing_total_is_partial() -> None:
    compare_path = f"{REPO_PATH}/compare/{BASE_SHA}...{C2}"
    client, _, _ = make_client({compare_path: [canned("compare_complete")]}, per_page=100)
    enumeration = client.compare_commits(OWNER, REPO, BASE_SHA, C2)
    assert enumeration.coverage == "complete_for_snapshot" and enumeration.reason is None and enumeration.shas == [C1, C2]

    body = canned_body("compare_complete")
    del body["total_commits"]
    client, _, _ = make_client({compare_path: [fixture_response(200, body)]}, per_page=100)
    enumeration = client.compare_commits(OWNER, REPO, BASE_SHA, C2)
    assert enumeration.coverage == "partial" and "total_commits" in (enumeration.reason or "")


@pytest.mark.parametrize("revision", ["--all", "a..b", "main branch", ""])
def test_compare_rejects_option_like_or_range_revisions(revision: str) -> None:
    client, transport, _ = make_client({})
    with pytest.raises(InputError) as info:
        client.compare_commits(OWNER, REPO, revision, C2)
    assert info.value.code == "INVALID_REVISION" and info.value.exit_code == 2
    assert transport.requests == []


def test_module_exports_no_write_helpers() -> None:
    exported = set(github_module.__all__)
    assert {"GitHubClient", "FixtureTransport", "UrllibTransport", "fixture_response", "github_state_from_pull"} <= exported
    assert not any(name.lower().startswith(("post", "put", "patch", "delete", "write")) for name in exported)
