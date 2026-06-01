"""Vector store tool.

Allows the agent to query the session's embedded document database.
"""
from agent.memory import Memory, _load_vectorstore
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

    # Lazily initialize the vector store on first use
    if mem.vectorstore is None:
        mem.vectorstore = _load_vectorstore(mem.session_dir)

    if mem.vectorstore is None:
        return {"error": "Vector store is not available. Check that chromadb and tiktoken are installed, and embedding config is correct."}

    query = args.get("query", "")
    top_k = args.get("top_k", 3)

    if not query:
        return {"error": "Query cannot be empty"}

    try:
        results = mem.vectorstore.query(query, top_k=top_k)
    except Exception as e:
        return {"error": f"Vector store query failed: {e}"}

    if not results:
        return {"results": [], "message": "No relevant documents found"}

    return {"results": results, "count": len(results)}
