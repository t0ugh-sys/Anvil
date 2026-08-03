# Anvil Skills

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

Skills are loaded via the interactive runtime's `--skill` flag, then used from within the chat session:

```bash
anvil --skill files --skill memory --provider mock --model mock-v3
```

## Skill Notes

- [files/SKILL.md](D:\workspace\Anvil\skills\files\SKILL.md)
- [commands/SKILL.md](D:\workspace\Anvil\skills\commands\SKILL.md)
- [memory/SKILL.md](D:\workspace\Anvil\skills\memory\SKILL.md)
- [web_search/SKILL.md](D:\workspace\Anvil\skills\web_search\SKILL.md)
- [browser/SKILL.md](D:\workspace\Anvil\skills\browser\SKILL.md)
