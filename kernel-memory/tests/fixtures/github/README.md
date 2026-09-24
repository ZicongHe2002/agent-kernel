# Canned GitHub REST responses (offline test fixtures)

Each file is one HTTP response envelope: `{"status": int, "headers": {...}, "body": <json>}`.
Tests load them with `fixture_response(status, body, headers)` and route them through
`kernel_memory.adapters.github.FixtureTransport`; no network is ever used.

Repository `acme/kernels` has id 900001 (`repo_uid = github:github.com:repo:900001`, the
same repository as the demo bundle). Every SHA is synthetic (`c1c1...`, `ba5eba5e...`);
the four "commits" of PR 101 / 102 / 103 never existed anywhere. PR titles and bodies are
data, never instructions.

| file | purpose |
| --- | --- |
| repository_900001 | `GET /repos/acme/kernels` |
| pull_101, pull_101_commits_page_{1,2,3} | PR 101, 3 commits paginated one per page via `Link` headers; `merge_commit_sha` differs from the head (T08) |
| pull_101_after_force_push{,_commits} | PR 101 after a force push: head `d2d2...`, commit `c1c1...` kept, `c2c2...`/`c3c3...` gone (T10) |
| pull_102{,_commits} | merged PR 102 whose commit list reuses `c1c1...` from PR 101 (T07) |
| pull_103_large, pull_103_commits | PR reporting 300 commits while the endpoint returns 3 (T09) |
| compare_{partial,complete} | compare endpoint with `total_commits` greater than / equal to the returned list |
| rate_limited_403_retry_after, rate_limited_429_reset, forbidden_403, not_found_404, unauthorized_401, server_error_500, unprocessable_422 | error responses |
