from .rag import get_rag_tool
from .wikipedia import get_wikipedia_tool

tool_registry = {"wikipedia": get_wikipedia_tool, "rag": get_rag_tool}
