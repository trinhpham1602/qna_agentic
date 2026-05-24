from __future__ import annotations

import asyncio
from typing import Any

from vietjet.qna_graph import AgenticState, build_graph

_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


async def ask(question: str) -> dict[str, Any]:
    graph = get_graph()
    init: AgenticState = {"question": question, "attempts": 0}
    return await graph.ainvoke(init)


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "Phí đổi vé Vietjet hạng eco quốc nội bao nhiêu"
    out = asyncio.run(ask(q))

    print("\n=== Question ===")
    print(q)
    print(
        f"\n=== Doc type: {out.get('doc_type')} | rewrites: {out.get('attempts', 0)} ==="
    )
    if out.get("web_skipped_reason"):
        print(f"\n=== Web skipped: {out['web_skipped_reason']} ===")
    else:
        print(
            f"\n=== Web: {len(out.get('web_candidates') or [])} candidates → "
            f"{len(out.get('web_chosen_urls') or [])} chosen → "
            f"{len(out.get('web_docs') or [])} fetched ==="
        )
    print(f"\n=== DB docs: {len(out.get('docs') or [])} ===")
    print("\n=== Answer ===")
    print(out.get("answer"))
    print("\n=== Citations ===")
    for c in out.get("citations") or []:
        print(f"  - {c}")
