from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from orchestration_agent import build_graph

app = FastAPI(
    title="LangGraph Multi-Agent Query System",
    description="Natural language interface for database and external search operations",
    version="1.0.0"
)

graph = build_graph()

class QueryRequest(BaseModel):
    query: str

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Show me all people in the database"
            }
        }


class QueryResponse(BaseModel):
    query: str
    intent: Optional[str]
    agent: Optional[str]
    response: Optional[dict]
    error: Optional[str]
    
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "multi-agent-system",
        "graph_compiled": graph is not None
    }

@app.post("/query")
def process_query(request: QueryRequest):
    
    query = request.query.strip()
    
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    print(f"\n{'='*60}")
    print(f"NEW QUERY: {query}")
    print(f"{'='*60}")
    
    try:
        state = {
            "query": query,
            "intent": None,
            "local_result": None,
            "external_result": None,
            "final_response": None,
            "error": None
        }

        result = graph.invoke(state)
        
        print(f"QUERY COMPLETED")
        print(f"Intent: {result.get('intent')}")
        print(f"Agent: {result.get('final_response', {}).get('agent', 'Unknown')}")

        return {
            "query": query,
            "intent": result.get("intent"),
            "response": result.get("final_response"),
            "error": result.get("error")
        }
        
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        print(f"ERROR: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/query/detailed")
def process_query_detailed(request: QueryRequest):

    query = request.query.strip()
    
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        state = {
            "query": query,
            "intent": None,
            "local_result": None,
            "external_result": None,
            "final_response": None,
            "error": None
        }

        result = graph.invoke(state)

        return {
            "query": query,
            "intent": result.get("intent"),
            "local_result": result.get("local_result"),
            "external_result": result.get("external_result"),
            "final_response": result.get("final_response"),
            "error": result.get("error"),
            "full_state": result
        }
        
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        raise HTTPException(status_code=500, detail=error_msg)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)