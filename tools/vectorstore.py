"""Vector store tool.

Allows the agent to query the session's embedded document database.
"""
from agent.memory import Memory
from agent.tools import tool


@tool(
    name="search_knowledge",
    description=(
        "Search the embedded document database for relevant information.\n"
        "Use this when the user asks about specific documents, files, "
        "or topics that might have been previously embedded using the /embed command."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to find relevant information.",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default: 3).",
            },
        },
        "required": ["query"],
    },
)
def search_knowledge_handler(args):
    """Search the vector store for relevant chunks."""
    mem = Memory()
    vs = mem.vectorstore
    
    query = args.get("query", "")
    top_k = args.get("top_k", 3)
    
    if not query:
        return {"error": "Query cannot be empty"}
        
    results = vs.query(query, top_k=top_k)
    
    if not results:
        return {"results": [], "message": "No relevant documents found"}
        
    return {"results": results, "count": len(results)}
