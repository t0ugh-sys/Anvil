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

- [files/SKILL.md](D:\workspace\LoopAgent\skills\files\SKILL.md)
- [commands/SKILL.md](D:\workspace\LoopAgent\skills\commands\SKILL.md)
- [memory/SKILL.md](D:\workspace\LoopAgent\skills\memory\SKILL.md)
- [web_search/SKILL.md](D:\workspace\LoopAgent\skills\web_search\SKILL.md)
- [browser/SKILL.md](D:\workspace\LoopAgent\skills\browser\SKILL.md)
