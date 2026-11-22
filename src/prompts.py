"""Load versioned prompts from prompts/ (the prompt library, Stage 3).

File names carry the id and version, e.g. prompts/build/PR02_answer_with_citations_v1.0.txt.
The newest version of an id is used unless a version is asked for; the id + version go into the
decision log (prompt_version) so every answer can be traced to the exact prompt text.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
_NAME = re.compile(r"^PR(\d+)_.*_v(\d+(?:\.\d+)*)\.txt$")


def _version_key(v: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in v.split("."))


@lru_cache(maxsize=32)
def load_prompt(prompt_id: str, version: Optional[str] = None) -> Tuple[str, str]:
    """Return (template_text, "PR-02 v1.0"). prompt_id like "PR-02"."""
    num = prompt_id.replace("PR-", "").zfill(2)
    found = []
    for p in PROMPT_DIR.rglob("*.txt"):
        m = _NAME.match(p.name)
        if m and m.group(1).zfill(2) == num and (version is None or m.group(2) == version):
            found.append((m.group(2), p))
    if not found:
        raise FileNotFoundError(f"Prompt {prompt_id} v{version or 'latest'} not found under {PROMPT_DIR}")
    ver, path = max(found, key=lambda x: _version_key(x[0]))
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines and lines[0].startswith("# PR-"):  # header comment, not part of the prompt
        lines = lines[1:]
    return "\n".join(lines).strip(), f"PR-{num} v{ver}"


def fill(template: str, **values: str) -> str:
    """Replace {name} placeholders only (JSON braces in the template are left alone)."""
    out = template
    for k, v in values.items():
        out = out.replace("{" + k + "}", v)
    return out


def neutralise_tags(text: str, *tags: str) -> str:
    """Stop customer text from closing or opening our delimiter tags (prompt injection, NFR-10)."""
    for tag in tags:
        text = re.sub(rf"</?\s*{tag}\b[^>]*>", f"[{tag} tag removed]", text, flags=re.IGNORECASE)
    return text
