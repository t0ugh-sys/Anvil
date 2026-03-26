---
name: web_search
description: Search the web and fetch URLs
---

# web_search

Use `web_search` when the task needs current external information that does not
exist in the local workspace.

Rules:

- Prefer local repository context first.
- Persist fetched findings into run traces or summaries when they affect plan or facts.
- Keep fetched content concise and actionable.
