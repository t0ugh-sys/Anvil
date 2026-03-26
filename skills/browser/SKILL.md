---
name: browser
description: Browser automation (requires playwright)
---

# browser

Use `browser` when the task needs interactive page automation rather than
simple HTTP fetches.

Provided capability:

- Playwright-backed browser automation

Rules:

- Keep browser work explicit and goal-directed.
- Prefer lighter tools such as `web_search` or `fetch_url` when automation is unnecessary.
- Treat browser support as optional and dependency-gated.
