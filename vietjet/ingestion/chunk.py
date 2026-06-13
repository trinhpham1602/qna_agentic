"""Stage 4: chunk normalized markdown into retrieval units.

  - Walk H1/H2/H3/H4 to maintain `section_path`.
  - Tables ≥ MIN_TABLE_ROWS become standalone chunks with section path as
    preamble — never split mid-table.
  - Remaining prose is greedy-packed under MAX_CHARS, splitting on
    paragraph boundaries with one-paragraph overlap.

Output: chunks.jsonl  (one JSON record per line).
"""

from __future__ import annotations
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from vietjet.config import CHUNKS_PATH, DOC_TYPE_MAP, NORM_DIR

MAX_CHARS = 1800
OVERLAP_CHARS = 200
MIN_TABLE_ROWS = 4

HEADER_RE = re.compile(r"^(#{1,4})\s+(.*?)\s*$")


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    doc_type: str
    section_path: str
    has_table: bool


def split_sections(md: str) -> list[tuple[str, str]]:
    path: list[str | None] = [None, None, None, None]
    sections: list[tuple[str, list[str]]] = []
    buf: list[str] = []

    def emit() -> None:
        nonlocal buf
        if any(line.strip() for line in buf):
            current = " > ".join(p for p in path if p)
            sections.append((current, "\n".join(buf).strip()))
        buf = []

    for line in md.splitlines():
        m = HEADER_RE.match(line)
        if m:
            emit()
            depth = len(m.group(1))
            title = m.group(2).strip()
            path[depth - 1] = title
            for i in range(depth, len(path)):
                path[i] = None
            buf = [line]
        else:
            buf.append(line)
    emit()
    return sections


def split_prose_and_tables(body: str) -> list[tuple[str, bool]]:
    segments: list[tuple[list[str], bool]] = []
    cur: list[str] = []
    cur_is_tbl = False
    for line in body.splitlines():
        is_tbl = line.lstrip().startswith("|")
        if is_tbl != cur_is_tbl and cur:
            segments.append((cur, cur_is_tbl))
            cur = []
        cur_is_tbl = is_tbl
        cur.append(line)
    if cur:
        segments.append((cur, cur_is_tbl))

    out: list[tuple[str, bool]] = []
    for seg, is_tbl in segments:
        text = "\n".join(seg).strip()
        if not text:
            continue
        tbl_rows = sum(1 for l in seg if l.lstrip().startswith("|"))
        out.append((text, is_tbl and tbl_rows >= MIN_TABLE_ROWS))
    return out


def pack_paragraphs(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for p in paras:
        plen = len(p) + 2
        if cur and cur_len + plen > max_chars:
            chunks.append("\n\n".join(cur).strip())
            tail = cur[-1]
            if len(tail) <= overlap_chars:
                cur = [tail]
                cur_len = len(tail) + 2
            else:
                cur = []
                cur_len = 0
        cur.append(p)
        cur_len += plen
    if cur:
        chunks.append("\n\n".join(cur).strip())
    return [c for c in chunks if c]


def chunk_file(path: Path) -> Iterable[Chunk]:
    stem = path.stem
    doc_type = DOC_TYPE_MAP.get(stem, "other")
    md = unicodedata.normalize("NFC", path.read_text(encoding="utf-8"))
    idx = 0
    for section_path, body in split_sections(md):
        if not body.strip():
            continue
        for segment, is_table in split_prose_and_tables(body):
            if is_table:
                preamble = f"[Mục: {section_path}]\n\n" if section_path else ""
                yield Chunk(
                    id=f"{stem}::{idx:03d}",
                    text=f"{preamble}{segment}".strip(),
                    source=stem,
                    doc_type=doc_type,
                    section_path=section_path,
                    has_table=True,
                )
                idx += 1
            else:
                for piece in pack_paragraphs(segment, MAX_CHARS, OVERLAP_CHARS):
                    yield Chunk(
                        id=f"{stem}::{idx:03d}",
                        text=piece,
                        source=stem,
                        doc_type=doc_type,
                        section_path=section_path,
                        has_table=False,
                    )
                    idx += 1


def chunk_folder(src: Path = NORM_DIR, out_path: Path = CHUNKS_PATH) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(src.glob("*.md")):
        file_chunks = list(chunk_file(path))
        chunks.extend(file_chunks)
        n_tbl = sum(1 for c in file_chunks if c.has_table)
        print(f"[CHUNK] {path.stem:60s} {len(file_chunks):3d} chunks ({n_tbl} table)")
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    print(f"\nTotal: {len(chunks)} chunks → {out_path}")
    return chunks


def main() -> None:
    chunk_folder()


if __name__ == "__main__":
    main()
