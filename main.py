import uvicorn
from fastapi import FastAPI
import uuid
from pydantic import BaseModel
from contextlib import asynccontextmanager
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from rag_workflow import builder, escalate_api_action, DB_CONNECTION_STRING

# Globals for the compiled graph
compiled_rag_graph = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global compiled_rag_graph
    # Initialize the async connection pool and checkpointer
    async with AsyncConnectionPool(DB_CONNECTION_STRING) as async_pool:
        checkpointer = AsyncPostgresSaver(async_pool)
        await checkpointer.setup()
    
        # Compile the graph globally
        compiled_rag_graph = builder.compile(checkpointer=checkpointer)
        img = compiled_rag_graph.get_graph().draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(img)
        yield
    
class ThreadResponse(BaseModel):
    thread_id: str

class ChatRequest(BaseModel):
    thread_id: str
    query: str

class ChatResponse(BaseModel):
    answer: str
    suggest_escalate: bool = False

app = FastAPI(
    title="Customer Service API",
    description="RAG + LangGraph Escalation Flow",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/thread", response_model=ThreadResponse)
async def create_thread_endpoint():
    """Tạo mới một phiên hội thoại (Thread ID)"""
    return ThreadResponse(thread_id=str(uuid.uuid4()))

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint to receive user query and route it through the RAG LangGraph.
    
    If the query involves high-risk situations (e.g. Hoàn tiền giá trị cao),
    the LangGraph will asynchronously escalate to a human and return early.
    Otherwise, it will generate an LLM response.
    """
    new_input = {
        "question": request.query
    }
    
    config = {"configurable": {"thread_id": request.thread_id}}
    
    # Run the compiled LangGraph async with checkpointer config
    final_state = await compiled_rag_graph.ainvoke(new_input, config=config)
    
    # If the system detected max risk level >= 3 (or any desired threshold),
    # explicitly signal the frontend to ask for user confirmation.
    suggest_escalate = final_state.get("max_risk_level", 0) >= 3
    
    return ChatResponse(
        answer=final_state["final_answer"], 
        suggest_escalate=suggest_escalate
    )

class EscalateResponse(BaseModel):
    status: str
    message: str

@app.post("/escalate", response_model=EscalateResponse)
async def escalate_endpoint():
    """
    Endpoint for UI to trigger a human escalation when the user confirms 
    the Suggest Escalation prompt.
    """
    message = await escalate_api_action()
    return EscalateResponse(status="success", message=message)

if __name__ == "__main__":
    # POSTMAN TEST INSTRUCTIONS
    # Start server: python main.py
    # Test curl:
    # curl -X 'POST' \
    #   'http://127.0.0.1:8000/chat' \
    #   -H 'accept: application/json' \
    #   -H 'Content-Type: application/json' \
    #   -d '{
    #   "query": "Tôi muốn trả lại vé"
    # }'
    
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
