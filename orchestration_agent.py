from typing import TypedDict, Optional, Any, Dict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
import os
from constants import GROQ_API_KEY, POSTGRES_URI
from agents import local_db_agent, external_search_agent, hybrid_agent

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY
)

class GraphState(TypedDict):
    query: str
    intent: Optional[str]
    local_result: Optional[Any]
    external_result: Optional[Any]
    final_response: Optional[Any]
    error: Optional[str]


def intent_classifier(state: GraphState) -> Dict[str, Any]:

    query = state["query"]
    
    # Comprehensive prompt for intent classification
    classification_prompt = f"""You are an intelligent intent classification system for a multi-agent recruitment database application.

SYSTEM OVERVIEW:
You have access to THREE specialized agents:

1. LOCAL_DB_AGENT
   - Purpose: Operates on existing PostgreSQL database
   - Capabilities: SELECT, INSERT (single record), UPDATE, DELETE
   - Use when: User wants to query/modify existing database records
   
2. EXTERNAL_SEARCH_AGENT
   - Purpose: Searches external recruitment platforms (LinkedIn, Indeed, Glassdoor)
   - Capabilities: Real-time candidate search using AI
   - Use when: User wants to find candidates NOT yet in database
   
3. HYBRID_AGENT
   - Purpose: Combines external search with database insertion
   - Capabilities: Search external sources AND automatically add results to database
   - Use when: User wants to find AND save candidates

CLASSIFICATION RULES:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rule 1: Classify as LOCAL if query involves:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Viewing/listing existing database records
✓ Counting/statistics on existing data
✓ Adding a SPECIFIC named person (not a search)
✓ Updating existing records
✓ Deleting records
✓ Filtering/querying existing data

Keywords: "show", "list", "display", "get from database", "how many", 
         "count", "add [specific name with details]", "update", "delete",
         "who are", "what are", "in our database", "from our team"

Examples:
✓ "Show me all people in the database"
✓ "How many engineers are in Mumbai?"
✓ "List all data scientists"
✓ "Add Vikram Desai as a DevOps Engineer from Delhi"
✓ "Update Rahul Patil's role to Senior Backend Engineer"
✓ "Delete Amit Sharma from database"
✓ "Who are the machine learning engineers in our team?"
✓ "Get all people from Pune"
✓ "Count people by location"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rule 2: Classify as EXTERNAL if query involves:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Searching for candidates from external sources
✓ Finding people NOT yet in database
✓ Looking up job candidates online
✓ NO mention of adding/saving to database

Keywords: "find", "search for", "look up", "get me", "discover",
         WITHOUT any mention of "add", "save", "insert", or "database"

Examples:
✓ "Find machine learning engineers in San Francisco"
✓ "Search for AI researchers in Europe"
✓ "Look up data scientists in Boston"
✓ "Get me frontend developers in Seattle"
✓ "Discover cloud architects in Singapore"
✓ "Find blockchain developers"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rule 3: Classify as HYBRID if query involves:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Searching external sources AND saving to database
✓ Finding candidates AND adding them to our team
✓ Explicitly mentions both searching and database operations

Keywords: "search AND add", "find AND save", "look up AND insert",
         "add top N", "save to database", "add to our database",
         "save them", "add to team"

Examples:
✓ "Search for AI researchers in Europe and add the top 5 to our database"
✓ "Find ML engineers in San Francisco and save them to database"
✓ "Look up data scientists in Boston and add top 3 to our team"
✓ "Search for full stack developers in Berlin and add them"
✓ "Find DevOps engineers and save the best ones to database"
✓ "Get blockchain developers and add top 5"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CURRENT USER QUERY:
"{query}"

TASK:
Analyze the query carefully and classify it into EXACTLY ONE category: LOCAL, EXTERNAL, or HYBRID

IMPORTANT:
- If the query mentions BOTH searching and adding/saving → HYBRID
- If the query ONLY mentions searching/finding → EXTERNAL  
- If the query operates on existing database → LOCAL
- When in doubt between EXTERNAL and HYBRID, check for keywords: "add", "save", "insert", "database"

OUTPUT:
Respond with EXACTLY ONE WORD (no explanation, no punctuation):
LOCAL
or
EXTERNAL
or
HYBRID"""

    try:
        print(f"\nORCHESTRATOR {'='*60}")
        print(f"ORCHESTRATOR Analyzing Query: '{query}'")
        print(f"ORCHESTRATOR {'='*60}")
        
        # Get LLM classification
        response = llm.invoke(classification_prompt)
        # intent = response.content.strip().upper()
        intent = (
            response.content.strip().upper()
            if hasattr(response, "content") and isinstance(response.content, str)
            else str(response).strip().upper()
        )

        
        # Validate and clean the intent
        valid_intents = ["LOCAL", "EXTERNAL", "HYBRID"]
        
        # Remove any extra words or punctuation
        for valid_intent in valid_intents:
            if valid_intent in intent:
                intent = valid_intent
                break
        
        # Final validation
        if intent not in valid_intents:
            print(f"ORCHESTRATOR  Invalid intent '{intent}', applying fallback logic...")
            
            # Fallback logic based on keywords
            query_lower = query.lower()
            
            if any(word in query_lower for word in ["add", "save", "insert"]) and \
               any(word in query_lower for word in ["find", "search", "look up"]):
                intent = "HYBRID"
                print(f"ORCHESTRATOR Fallback: Detected search + add keywords → HYBRID")
            elif any(word in query_lower for word in ["find", "search", "look up", "discover"]):
                intent = "EXTERNAL"
                print(f"ORCHESTRATOR Fallback: Detected search keywords → EXTERNAL")
            else:
                intent = "LOCAL"
                print(f"ORCHESTRATOR Fallback: Default to LOCAL")
        
        # Log the decision
        print(f"ORCHESTRATOR ✓ Intent Classified: {intent}")
        print(f"ORCHESTRATOR → Routing to: {intent}_AGENT")
        print(f"ORCHESTRATOR {'='*60}\n")
        
        return {"intent": intent}
        
    except Exception as e:
        error_msg = f"Intent classification failed: {str(e)}"
        print(f"ORCHESTRATOR ERROR: {error_msg}")
        return {"intent": "ERROR", "error": error_msg}


def route_by_intent(state: GraphState) -> str:
    """
    Routes execution to the appropriate agent based on classified intent
    
    This function acts as a traffic controller, directing the request
    to the correct specialized agent.
    """
    intent = state.get("intent", "ERROR")
    
    routing_map = {
        "LOCAL": "local_db_agent",
        "EXTERNAL": "external_search_agent",
        "HYBRID": "hybrid_agent",
        "ERROR": "error_handler"
    }
    
    route = routing_map.get(intent, "error_handler")
    
    print(f"ORCHESTRATOR 🔀 Routing Decision: {intent} → {route}")
    
    return route

def error_handler(state: GraphState) -> Dict[str, Any]:
    """
    Handles errors in the workflow
    
    This agent is called when something goes wrong in intent classification
    or when an invalid intent is detected.
    """
    error_msg = state.get("error", "Unknown error occurred in the system")
    query = state.get("query", "Unknown query")
    
    print(f"Error encountered")
    print(f"ERROR_HANDLER Query: {query}")
    print(f"ERROR_HANDLER Error: {error_msg}")
    
    return {
        "error": error_msg,
        "final_response": {
            "agent": "ERROR_HANDLER",
            "status": "failed",
            "message": "Unable to process the request. Please try rephrasing your query.",
            "error_details": error_msg,
            "suggestions": [
                "Make sure your query is clear and specific",
                "For database queries: 'Show me all engineers'",
                "For external search: 'Find AI researchers in Europe'", 
                "For hybrid: 'Search for ML engineers and add top 5 to database'"
            ]
        }
    }

def build_graph():    
    
    workflow = StateGraph(GraphState)
    print("ORCHESTRATOR Adding nodes:")
    
    workflow.add_node("intent_classifier", intent_classifier)
    print("ORCHESTRATOR intent_classifier - Orchestration & routing")
    
    workflow.add_node("local_db_agent", local_db_agent)
    print("ORCHESTRATOR local_db_agent - Database operations")
    
    workflow.add_node("external_search_agent", external_search_agent)
    print("ORCHESTRATOR external_search_agent - External candidate search")
    
    workflow.add_node("hybrid_agent", hybrid_agent)
    print("ORCHESTRATOR hybrid_agent - Search + Database insert")
    
    workflow.add_node("error_handler", error_handler)
    print("ORCHESTRATOR error_handler - Error management")

    # Set entry point - always start with intent classification
    workflow.set_entry_point("intent_classifier")
    print("ORCHESTRATOR Entry point: intent_classifier")

    # Add conditional routing based on intent
    # This is where the orchestration magic happens
    workflow.add_conditional_edges(
        "intent_classifier",  # Source node
        route_by_intent,      # Routing function
        {
            "local_db_agent": "local_db_agent",
            "external_search_agent": "external_search_agent",
            "hybrid_agent": "hybrid_agent",
            "error_handler": "error_handler"
        }
    )
    print("ORCHESTRATOR Conditional routing configured")

    # All agents terminate the workflow (no further processing)
    workflow.add_edge("local_db_agent", END)
    workflow.add_edge("external_search_agent", END)
    workflow.add_edge("hybrid_agent", END)
    workflow.add_edge("error_handler", END)
    print("ORCHESTRATOR Terminal nodes configured")
    
    print("ORCHESTRATOR Workflow built successfully!")
    
    print("\nORCHESTRATOR System Summary:")
    print("ORCHESTRATOR   • Orchestration: Intent-based routing")
    print("ORCHESTRATOR   • Agent 1: LOCAL_DB_AGENT (Database operations)")
    print("ORCHESTRATOR   • Agent 2: EXTERNAL_SEARCH_AGENT (Candidate search)")
    print("ORCHESTRATOR   • Agent 3: HYBRID_AGENT (Search + Insert)")
    print("ORCHESTRATOR   • Error Handler: Graceful error management")
    print("ORCHESTRATOR " + "="*60 + "\n")
    
    # Compile and return the graph
    compiled_graph = workflow.compile()
    print("ORCHESTRATOR Graph compiled and ready for execution!\n")
    

    return compiled_graph
