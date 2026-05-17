"""Langgraph-driven agentic Q&A over Vietjet docs.

Flow:
  route → retrieve → grade → (rewrite ⤴ retrieve, ≤ MAX_REWRITES) → generate

  • route       Classify intent into a doc_type (pricing/baggage/…) and decide
                whether to boost table chunks.
  • retrieve    Hybrid pgvector + BM25 + cross-encoder rerank, filtered by
                doc_type from the router.
  • grade       LLM judges whether the retrieved chunks are sufficient to
                answer. Returns "sufficient" or "insufficient".
  • rewrite     If grade is insufficient and we have rewrite budget, LLM
                rewrites the query to widen retrieval, then loops to retrieve.
                The doc_type filter is dropped on the second pass to avoid
                routing-error lock-in.
  • generate    Produces the final Vietnamese answer with [source#section]
                citations.
"""

from __future__ import annotations
import re
from typing import Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from vietjet.config import LLM_MODEL, MAX_REWRITES, TOP_K
from vietjet.retriever import get_retriever

# ------------------------------------------------------------------
# State
# ------------------------------------------------------------------

DocType = Literal[
    "regulation", "pricing", "baggage", "procedure", "service",
    "compensation", "payment", "other",
]


class AgentState(TypedDict, total=False):
    question: str            # original
    query: str               # current, possibly rewritten
    doc_type: str | None     # router decision; None disables metadata filter
    boost_tables: bool
    docs: list[Document]
    attempts: int            # rewrite count
    sufficient: bool
    answer: str
    citations: list[str]


# ------------------------------------------------------------------
# LLM
# ------------------------------------------------------------------

_llm: ChatOllama | None = None


def _get_llm(temperature: float = 0.0) -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = ChatOllama(model=LLM_MODEL, temperature=temperature)
    return _llm


# ------------------------------------------------------------------
# Routing (light, keyword + LLM fallback)
# ------------------------------------------------------------------

_KW_TYPE: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(phí|lệ phí|giá|tiền|cost|vnd|usd|hóa đơn|vat)\b", re.I), "pricing"),
    (re.compile(r"\b(hành lý|baggage|kg|ký gửi|xách tay|quá khổ|đồ cấm|lithium|pin)\b", re.I), "baggage"),
    (re.compile(r"\b(bồi thường|delay|chậm|hủy chuyến|đền bù)\b", re.I), "compensation"),
    (re.compile(r"\b(cmnd|passport|hộ chiếu|giấy tờ|trẻ em|um|sky kids|thú cưng|sky pet)\b", re.I), "procedure"),
    (re.compile(r"\b(check[- ]?in|nối chuyến|thủ tục|làm thủ tục)\b", re.I), "procedure"),
    (re.compile(r"\b(thanh toán|momo|vnpay|hd saison|thẻ tín dụng|kênh)\b", re.I), "payment"),
    (re.compile(r"\b(skyboss|business|hạng thương gia|lounge|phòng chờ|deluxe|eco)\b", re.I), "service"),
    (re.compile(r"\b(điều kiện vé|điều lệ|hoàn vé|đổi vé|bảo lưu)\b", re.I), "regulation"),
]

_TABLE_HINT = re.compile(r"\b(bao nhiêu|giá|phí|kg|bảng|tỷ giá|mức)\b", re.I)


async def route_node(state: AgentState) -> AgentState:
    q = state["question"]
    doc_type: str | None = None
    for pat, t in _KW_TYPE:
        if pat.search(q):
            doc_type = t
            break
    return {
        "query": q,
        "doc_type": doc_type,
        "boost_tables": bool(_TABLE_HINT.search(q)),
        "attempts": 0,
    }


# ------------------------------------------------------------------
# Retrieve
# ------------------------------------------------------------------

async def retrieve_node(state: AgentState) -> AgentState:
    retriever = get_retriever(use_rerank=True)
    # After the first rewrite, drop the metadata filter — keep us from being
    # locked into a wrong routing decision.
    doc_type = state.get("doc_type") if state.get("attempts", 0) == 0 else None
    docs = retriever.search(
        state["query"],
        top_k=TOP_K,
        doc_type=doc_type,
        boost_tables=state.get("boost_tables", False),
    )
    return {"docs": docs}


# ------------------------------------------------------------------
# Grade
# ------------------------------------------------------------------

_GRADE_PROMPT = """Bạn là người chấm chất lượng truy hồi tài liệu.

CÂU HỎI:
{question}

TÀI LIỆU TRUY HỒI:
{context}

Hãy đánh giá: các tài liệu trên có đủ thông tin để trả lời chính xác câu hỏi không?
Chỉ trả về MỘT TỪ DUY NHẤT:
- "yes" — nếu tài liệu có thông tin cụ thể (số liệu, điều khoản, quy định) cần thiết
- "no" — nếu thiếu hoặc chỉ liên quan mơ hồ
"""


def _format_context(docs: list[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        m = d.metadata
        parts.append(f"[{i}] {m['source']} > {m['section_path']}\n{d.page_content[:800]}")
    return "\n\n".join(parts)


async def grade_node(state: AgentState) -> AgentState:
    llm = _get_llm(temperature=0.0)
    msg = await llm.ainvoke([
        SystemMessage(content="Bạn là retrieval grader. Trả lời ngắn gọn."),
        HumanMessage(content=_GRADE_PROMPT.format(
            question=state["question"],
            context=_format_context(state["docs"]),
        )),
    ])
    verdict = (msg.content or "").strip().lower()
    return {"sufficient": verdict.startswith("y")}


# ------------------------------------------------------------------
# Rewrite
# ------------------------------------------------------------------

_REWRITE_PROMPT = """Câu hỏi gốc của khách hàng (tiếng Việt, có thể không chuẩn):
{question}

Tài liệu đã truy hồi KHÔNG đủ trả lời. Hãy viết lại câu hỏi để truy hồi tốt hơn:
- Thêm từ khóa kỹ thuật (tên dịch vụ, loại vé, đường bay, đơn vị tiền tệ, kg…)
- Bỏ filler/lịch sự
- Một câu duy nhất, ≤ 30 từ
- Không hỏi lại khách

Câu truy vấn mới:"""


async def rewrite_node(state: AgentState) -> AgentState:
    llm = _get_llm(temperature=0.2)
    msg = await llm.ainvoke([
        SystemMessage(content="Bạn là query rewriter cho hệ thống RAG."),
        HumanMessage(content=_REWRITE_PROMPT.format(question=state["question"])),
    ])
    new_query = (msg.content or "").strip().splitlines()[0].strip(' "“”')
    return {
        "query": new_query or state["question"],
        "attempts": state.get("attempts", 0) + 1,
    }


# ------------------------------------------------------------------
# Generate
# ------------------------------------------------------------------

_ANSWER_PROMPT = """Bạn là trợ lý CSKH của Vietjet Air. Trả lời câu hỏi dựa CHỈ trên tài liệu được cung cấp.

QUY TẮC:
- Trả lời bằng tiếng Việt, ngắn gọn, đúng trọng tâm
- Nếu có con số / điều khoản, trích chính xác
- Sau câu trả lời, liệt kê nguồn dạng: [nguồn: <source>#<section_path>]
- Nếu tài liệu KHÔNG đủ thông tin, nói rõ "Tôi không tìm thấy thông tin này trong tài liệu hiện có" và đề xuất liên hệ tổng đài 1900 1886

CÂU HỎI:
{question}

TÀI LIỆU:
{context}

TRẢ LỜI:"""


async def generate_node(state: AgentState) -> AgentState:
    llm = _get_llm(temperature=0.0)
    msg = await llm.ainvoke([
        SystemMessage(content="Bạn là Vietjet customer service assistant."),
        HumanMessage(content=_ANSWER_PROMPT.format(
            question=state["question"],
            context=_format_context(state["docs"]),
        )),
    ])
    citations = [f"{d.metadata['source']}#{d.metadata['section_path']}" for d in state["docs"]]
    return {"answer": (msg.content or "").strip(), "citations": citations}


# ------------------------------------------------------------------
# Routing edges
# ------------------------------------------------------------------

def _after_grade(state: AgentState) -> str:
    if state.get("sufficient"):
        return "generate"
    if state.get("attempts", 0) < MAX_REWRITES:
        return "rewrite"
    return "generate"


# ------------------------------------------------------------------
# Build graph
# ------------------------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("route", route_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade", grade_node)
    g.add_node("rewrite", rewrite_node)
    g.add_node("generate", generate_node)

    g.add_edge(START, "route")
    g.add_edge("route", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", _after_grade, {
        "generate": "generate",
        "rewrite": "rewrite",
    })
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", END)
    compiled_graph = g.compile()
    img = compiled_graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(img)
    return compiled_graph


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


async def ask(question: str) -> dict:
    graph = get_graph()
    return await graph.ainvoke({"question": question})


if __name__ == "__main__":
    import asyncio
    import sys

    q = " ".join(sys.argv[1:]) or "Qui dinh ve phu nu mang thai"
    out = asyncio.run(ask(q))
    print("\n=== Question ===")
    print(q)
    print(f"\n=== Doc type routed: {out.get('doc_type')}  | rewrites: {out.get('attempts', 0)} ===")
    print("\n=== Answer ===")
    print(out["answer"])
    print("\n=== Citations ===")
    for c in out.get("citations", []):
        print(f"  - {c}")
