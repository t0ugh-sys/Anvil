from __future__ import annotations

from loop_agent.skills import SkillLoader, list_skills


def main() -> None:
    loader = SkillLoader()
    for name in ('files', 'memory', 'commands'):
        loader.load(name)

    print(f'available={sorted(list_skills())}')
    print(f'loaded={loader.list_loaded()}')
    print(f'tool_names={sorted(loader.get_tools().keys())}')


if __name__ == '__main__':
    main()
