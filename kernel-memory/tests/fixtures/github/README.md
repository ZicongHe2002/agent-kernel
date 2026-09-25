# 预设的 GitHub REST 响应（离线测试夹具）

每个文件封装一个 HTTP 响应：`{"status": int, "headers": {...}, "body": <json>}`。
测试通过 `fixture_response(status, body, headers)` 加载这些响应，并由
`kernel_memory.adapters.github.FixtureTransport` 分发；整个过程不使用网络。

仓库 `acme/kernels` 的 ID 为 900001（`repo_uid = github:github.com:repo:900001`，
与演示数据包中的仓库相同）。所有 SHA 都是合成的（`c1c1...`、`ba5eba5e...`）；
PR 101 / 102 / 103 中的四个“提交”从未实际存在。PR 标题和正文始终作为数据处理，绝不作为指令。

| 文件 | 用途 |
| --- | --- |
| repository_900001 | `GET /repos/acme/kernels` |
| pull_101, pull_101_commits_page_{1,2,3} | PR 101 的 3 个提交通过 `Link` 响应头分页，每页 1 个；`merge_commit_sha` 与分支头提交不同（T08） |
| pull_101_after_force_push{,_commits} | 强制推送后的 PR 101：分支头为 `d2d2...`，保留提交 `c1c1...`，移除 `c2c2...` / `c3c3...`（T10） |
| pull_102{,_commits} | 已合并的 PR 102，其提交列表复用了 PR 101 中的 `c1c1...`（T07） |
| pull_103_large, pull_103_commits | PR 声称包含 300 个提交，但接口仅返回 3 个（T09） |
| compare_{partial,complete} | compare 接口中的 `total_commits` 分别大于或等于返回列表的长度 |
| rate_limited_403_retry_after, rate_limited_429_reset, forbidden_403, not_found_404, unauthorized_401, server_error_500, unprocessable_422 | 错误响应 |
