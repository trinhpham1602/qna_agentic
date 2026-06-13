"""Stage 2: keep only content between `![footer horizontal]` and `* * *`.

Vietjet pages wrap real content between a footer-image marker and a markdown
horizontal rule. Everything outside that window is header/nav/trailing junk.
"""

from __future__ import annotations
import re
from pathlib import Path

from vietjet.config import CLEAN_DIR, RAW_DIR

FOOTER_MARKER = re.compile(r"^\s*!\[footer horizontal\]")
HR_MARKER = re.compile(r"^\s*(?:\*\s*){3,}\s*$|^\s*\*\*\*\s*$")


def clean_markdown(markdown: str) -> str | None:
    lines = markdown.splitlines()
    start = next((i for i, line in enumerate(lines) if FOOTER_MARKER.match(line)), None)
    if start is None:
        return None
    end = next(
        (i for i, line in enumerate(lines[start + 1:], start=start + 1) if HR_MARKER.match(line)),
        len(lines),
    )
    body = "\n".join(lines[start + 1:end]).strip()
    return body or None


def clean_folder(src: Path = RAW_DIR, dst: Path = CLEAN_DIR) -> list[Path]:
    dst.mkdir(parents=True, exist_ok=True)
    cleaned: list[Path] = []
    for src_path in sorted(src.glob("*.md")):
        body = clean_markdown(src_path.read_text(encoding="utf-8"))
        if not body:
            print(f"[SKIP]  {src_path.name}: no footer/hr markers")
            continue
        out_path = dst / src_path.name
        out_path.write_text(body, encoding="utf-8")
        print(f"[CLEAN] {src_path.name} → {out_path}")
        cleaned.append(out_path)
    return cleaned


def main() -> None:
    clean_folder()


if __name__ == "__main__":
    main()
