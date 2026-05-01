from __future__ import annotations
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """
    Simple in-memory knowledge graph built from a triplet text file.

    Each non-comment line must be exactly: <from_id> <relation_id> <to_id>
    Lines starting with '#' are ignored.
    """

    def __init__(self) -> None:
        self.nodes: set[str] = set()
        self.edges: list[tuple[str, str, str]] = []          # (from, relation, to)
        self._adj: dict[str, list[dict]] = {}                 # from → [{to, relation}]
        self._reverse_adj: dict[str, list[dict]] = {}         # to   → [{from, relation}]

    @classmethod
    def from_dataset(cls, path: str | Path) -> "KnowledgeGraph":
        kg = cls()
        path = Path(path)
        if not path.exists():
            logger.warning("claude_dataset not found at %s — empty graph", path)
            return kg

        skipped = 0
        with path.open(encoding="utf-8") as f:
            for lineno, raw in enumerate(f, 1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 3:
                    logger.warning("Skipping malformed line %d: %r", lineno, raw.rstrip())
                    skipped += 1
                    continue
                from_id, relation_id, to_id = parts
                kg.add_edge(from_id, relation_id, to_id)

        logger.info(
            "KnowledgeGraph loaded: %d nodes, %d edges%s",
            len(kg.nodes),
            len(kg.edges),
            f" ({skipped} lines skipped)" if skipped else "",
        )
        return kg

    def add_edge(self, from_id: str, relation_id: str, to_id: str) -> None:
        self.nodes.add(from_id)
        self.nodes.add(to_id)
        self.edges.append((from_id, relation_id, to_id))
        self._adj.setdefault(from_id, []).append({"to": to_id, "relation": relation_id})
        self._reverse_adj.setdefault(to_id, []).append({"from": from_id, "relation": relation_id})

    def neighbors(self, node_id: str) -> list[dict]:
        """Return outgoing edges: [{to, relation}, ...]"""
        return self._adj.get(node_id, [])

    def predecessors(self, node_id: str) -> list[dict]:
        """Return incoming edges: [{from, relation}, ...]"""
        return self._reverse_adj.get(node_id, [])

    def traverse(
        self,
        seed_ids: set[str],
        depth: int = 2,
        relation_filter: set[str] | None = None,
    ) -> list[tuple[str, str, str]]:
        """
        BFS traversal from seed_ids up to `depth` hops.
        Returns list of (from_id, relation_id, to_id) tuples.
        Optional relation_filter keeps only edges with matching relation_id.
        """
        visited_edges: set[tuple[str, str, str]] = set()
        result: list[tuple[str, str, str]] = []
        frontier = set(seed_ids)

        for _ in range(depth):
            next_frontier: set[str] = set()
            for node in frontier:
                for edge in self._adj.get(node, []):
                    if relation_filter and edge["relation"] not in relation_filter:
                        continue
                    key = (node, edge["relation"], edge["to"])
                    if key not in visited_edges:
                        visited_edges.add(key)
                        result.append(key)
                        next_frontier.add(edge["to"])
            frontier = next_frontier - {e[0] for e in result}

        return result

    def __len__(self) -> int:
        return len(self.edges)

    def __repr__(self) -> str:
        return f"KnowledgeGraph(nodes={len(self.nodes)}, edges={len(self.edges)})"
