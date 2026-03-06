from __future__ import annotations

import pathlib
import re


REPLACEMENTS = [
    # builtins generics -> typing equivalents
    (re.compile(r"\bdict\["), "Dict["),
    (re.compile(r"\blist\["), "List["),
    (re.compile(r"\btuple\["), "Tuple["),
    (re.compile(r"\bset\["), "Set["),
    (re.compile(r"\bfrozenset\["), "FrozenSet["),
]


def rewrite_text(text: str) -> tuple[str, bool]:
    original = text

    for pattern, repl in REPLACEMENTS:
        text = pattern.sub(repl, text)

    # PEP604 unions: X | None -> Optional[X]
    text = re.sub(r"\b([A-Za-z_][A-Za-z0-9_\[\], ]*)\s*\|\s*None\b", r"Optional[\1]", text)
    # None | X -> Optional[X]
    text = re.sub(r"\bNone\s*\|\s*([A-Za-z_][A-Za-z0-9_\[\], ]*)\b", r"Optional[\1]", text)

    # Remaining unions: A | B -> Union[A, B]  (best-effort, simple cases only)
    def _union_repl(m: re.Match[str]) -> str:
        left = m.group(1).strip()
        right = m.group(2).strip()
        # Avoid rewriting bitwise ops by requiring type-ish tokens.
        return f"Union[{left}, {right}]"

    text = re.sub(
        r"\b([A-Za-z_][A-Za-z0-9_\.\[\], ]*)\s*\|\s*([A-Za-z_][A-Za-z0-9_\.\[\], ]*)\b",
        _union_repl,
        text,
    )

    return text, text != original


def ensure_typing_imports(text: str) -> str:
    needed = set()
    for name in ("Dict", "List", "Tuple", "Set", "FrozenSet", "Optional", "Union"):
        if re.search(rf"\b{name}\[", text) or re.search(rf"\b{name}\b", text) and name in ("Optional", "Union"):
            needed.add(name)

    if not needed:
        return text

    # If already imports typing, extend it.
    m = re.search(r"^from typing import ([^\n]+)$", text, flags=re.M)
    if m:
        existing = [p.strip() for p in m.group(1).split(",")]
        merged = sorted(set(existing) | needed)
        return text[: m.start(0)] + f"from typing import {', '.join(merged)}" + text[m.end(0) :]

    # Otherwise, insert after future import if present, else after module docstring if present, else at top.
    lines = text.splitlines(True)
    insert_at = 0

    if lines and lines[0].startswith("from __future__ import "):
        insert_at = 1
    elif lines and lines[0].startswith('"""'):
        # Skip module docstring
        for i in range(1, len(lines)):
            if lines[i].startswith('"""'):
                insert_at = i + 1
                break

    lines.insert(insert_at, f"from typing import {', '.join(sorted(needed))}\n")
    return "".join(lines)


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[1]
    targets = [*root.joinpath("src").rglob("*.py"), *root.joinpath("tests").rglob("*.py")]

    changed = 0
    for path in targets:
        text = path.read_text(encoding="utf-8")
        new_text, did = rewrite_text(text)
        if not did:
            continue
        new_text = ensure_typing_imports(new_text)
        path.write_text(new_text, encoding="utf-8")
        changed += 1

    print(f"updated_files={changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
