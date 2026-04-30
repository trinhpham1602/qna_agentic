"""
Visualize dataset/knowledge_graph.json as a PNG using networkx + matplotlib.
Run: python visualize_kg.py
Output: kg_visualization.png
"""
import json
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

KG_PATH  = Path("dataset/knowledge_graph.json")
OUT_PATH = Path("kg_visualization.png")

NODE_COLORS: dict[str, str] = {
    "airline":              "#4A90E2",
    "cabin_class":          "#27AE60",
    "route":                "#F39C12",
    "service":              "#8E44AD",
    "passenger_type":       "#E74C3C",
    "intent":               "#F1C40F",
    "reissue_type":         "#1ABC9C",
    "payment_method":       "#00B4D8",
    "payment_issue_type":   "#FF6B6B",
    "action":               "#7F8C8D",
}

EDGE_STYLES: dict[str, dict] = {
    "BELONGS_TO":  {"color": "#BDC3C7", "style": "dotted",  "width": 1.0},
    "HAS_POLICY":  {"color": "#3498DB", "style": "solid",   "width": 1.5},
    "HAS_TYPE":    {"color": "#1ABC9C", "style": "solid",   "width": 2.0},
    "RESTRICTS":   {"color": "#E74C3C", "style": "solid",   "width": 2.0},
    "ALLOWS":      {"color": "#2ECC71", "style": "solid",   "width": 2.0},
    "AFFECTS":     {"color": "#F39C12", "style": "dashed",  "width": 1.5},
    "REQUIRES":    {"color": "#9B59B6", "style": "solid",   "width": 1.5},
    "LEADS_TO":    {"color": "#00BCD4", "style": "solid",   "width": 1.5},
    "RESOLVES_VIA":{"color": "#C0392B", "style": "dashed",  "width": 1.5},
}


def main():
    with open(KG_PATH) as f:
        kg = json.load(f)

    G = nx.DiGraph()

    node_meta: dict[str, dict] = {}
    for node in kg["nodes"]:
        G.add_node(node["id"])
        node_meta[node["id"]] = node

    for edge in kg["edges"]:
        G.add_edge(edge["from"], edge["to"], edge_type=edge["type"])

    pos = nx.spring_layout(G, k=3.5, seed=42, iterations=60)

    fig, ax = plt.subplots(figsize=(28, 20))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.axis("off")

    # Draw edges grouped by type
    for etype, style in EDGE_STYLES.items():
        edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == etype]
        if not edges:
            continue
        nx.draw_networkx_edges(
            G, pos, edgelist=edges,
            edge_color=style["color"],
            style=style["style"],
            width=style["width"],
            arrows=True,
            arrowsize=12,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.1",
            ax=ax,
            alpha=0.75,
        )

    # Draw nodes grouped by type
    for ntype, color in NODE_COLORS.items():
        nodes = [n for n, d in node_meta.items() if d.get("type") == ntype]
        if not nodes:
            continue
        size = 600 if ntype in ("intent", "reissue_type") else 500
        nx.draw_networkx_nodes(
            G, pos, nodelist=nodes,
            node_color=color,
            node_size=size,
            ax=ax,
            alpha=0.95,
        )

    # Labels
    labels = {n: d.get("label", n) for n, d in node_meta.items()}
    nx.draw_networkx_labels(G, pos, labels, font_size=6.5, font_color="white", ax=ax)

    # Node-type legend
    node_patches = [
        mpatches.Patch(color=c, label=t.replace("_", " ").title())
        for t, c in NODE_COLORS.items()
    ]
    leg1 = ax.legend(
        handles=node_patches, title="Node types",
        loc="upper left", fontsize=8, title_fontsize=9,
        facecolor="#161b22", edgecolor="#30363d", labelcolor="white",
    )
    leg1.get_title().set_color("white")
    ax.add_artist(leg1)

    # Edge-type legend
    edge_patches = [
        mpatches.Patch(color=s["color"], label=t)
        for t, s in EDGE_STYLES.items()
    ]
    leg2 = ax.legend(
        handles=edge_patches, title="Edge types",
        loc="lower left", fontsize=8, title_fontsize=9,
        facecolor="#161b22", edgecolor="#30363d", labelcolor="white",
    )
    leg2.get_title().set_color("white")

    node_count = G.number_of_nodes()
    edge_count = G.number_of_edges()
    ax.set_title(
        f"OTA Flight Knowledge Graph  |  {node_count} nodes · {edge_count} edges",
        color="white", fontsize=14, fontweight="bold", pad=16,
    )

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    print(f"Saved: {OUT_PATH}  ({node_count} nodes, {edge_count} edges)")


if __name__ == "__main__":
    main()
