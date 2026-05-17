"""Interactive CLI for the Vietjet agentic Q&A."""

from __future__ import annotations
import asyncio
import sys

from vietjet.agent import ask


def _print_result(question: str, state: dict) -> None:
    print(f"\n— Routed doc_type: {state.get('doc_type')}  | rewrites: {state.get('attempts', 0)}")
    if state.get("attempts"):
        print(f"— Rewritten query: {state.get('query')!r}")
    print("\n" + state["answer"])
    cites = state.get("citations") or []
    if cites:
        print("\nNguồn:")
        for c in cites:
            print(f"  • {c}")


async def _repl() -> None:
    print("Vietjet Q&A — gõ câu hỏi rồi Enter. /q để thoát.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            continue
        if q in {"/q", "/quit", "/exit"}:
            return
        state = await ask(q)
        _print_result(q, state)


async def _oneshot(question: str) -> None:
    state = await ask(question)
    _print_result(question, state)


def main() -> None:
    args = sys.argv[1:]
    if args:
        asyncio.run(_oneshot(" ".join(args)))
    else:
        asyncio.run(_repl())


if __name__ == "__main__":
    main()
