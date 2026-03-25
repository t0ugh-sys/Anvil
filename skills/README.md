# LoopAgent Skills

This directory documents the skill layer in the same spirit as
`learn-claude-code`: a capability is more than a tool function. It includes the
scope, expectations, and safe operating boundaries.

## Built-in Skills

- `files`: read, write, patch, and search inside the workspace
- `commands`: execute local commands inside the workspace
- `memory`: inspect prior runs and summaries
- `web_search`: fetch public web content through the stdlib tool layer
- `browser`: optional Playwright-backed browser automation

## Loading Skills

```bash
python -m loop_agent.agent_cli skills
python -m loop_agent.agent_cli code --goal "inspect the repo" --workspace . --skill files --skill memory --provider mock --model mock-v3
```

## Skill Notes

- [files.md](D:\workspace\LoopAgent\skills\files.md)
- [commands.md](D:\workspace\LoopAgent\skills\commands.md)
- [memory.md](D:\workspace\LoopAgent\skills\memory.md)
- [web_search.md](D:\workspace\LoopAgent\skills\web_search.md)
