r"""Stage 3: promote bold `**Điều N.**`, `**N.M.**` patterns to ##/### headers.

Vietjet pages use bold for section structure instead of proper markdown headers.
A standard header splitter cannot detect them, so we promote those patterns
to actual `##` / `###` / `####` before chunking. Also normalizes NFD → NFC
because the source uses decomposed Unicode that breaks regex written in NFC.
"""

from __future__ import annotations
import re
import unicodedata
from pathlib import Path

from vietjet.config import CLEAN_DIR, NORM_DIR

DIEU_RE = re.compile(r"^\*\*\s*(Điều\s+\d+\..*?)\s*\*\*\s*$", re.MULTILINE)
DEEP_RE = re.compile(r"^\*\*\s*(\d+\.\d+\.\d+)\.?\s*([^*\n]+?)\s*\*\*\s*$", re.MULTILINE)
SUBSEC_RE = re.compile(r"^\*\*\s*(\d+\.\d+)\.?\s*([^*\n]+?)\s*\*\*\s*$", re.MULTILINE)
TOP_NUM_RE = re.compile(r"^\*\*\s*(\d+)\\?\.\s+([^*\n]+?)\s*\*\*\s*$", re.MULTILINE)
IMG_LINE_RE = re.compile(r"^\s*!\[[^\]]*\]\([^)]*\)\s*$", re.MULTILINE)


def normalize(md: str) -> str:
    md = unicodedata.normalize("NFC", md)
    md = md.replace("<Base64-Image-Removed>", "")
    md = IMG_LINE_RE.sub("", md)
    md = DEEP_RE.sub(r"#### \1 \2", md)
    md = SUBSEC_RE.sub(r"### \1 \2", md)
    md = DIEU_RE.sub(r"## \1", md)
    md = TOP_NUM_RE.sub(r"## \1. \2", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip() + "\n"


def normalize_folder(src: Path = CLEAN_DIR, dst: Path = NORM_DIR) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for src_path in sorted(src.glob("*.md")):
        text = normalize(src_path.read_text(encoding="utf-8"))
        out_path = dst / src_path.name
        out_path.write_text(text, encoding="utf-8")
        h2 = text.count("\n## ")
        h3 = text.count("\n### ")
        h4 = text.count("\n#### ")
        print(f"[NORM]  {src_path.name}  h2={h2} h3={h3} h4={h4}")


def main() -> None:
    normalize_folder()


if __name__ == "__main__":
    main()
