from genai_factory.config import WorkflowServerConfig, get_vector_db
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import Tool


def get_rag_tool(
    config: WorkflowServerConfig,
    collection: str,
    name: str,
    description: str = None,
    **kwargs
) -> Tool:

    vector_db = get_vector_db(config, collection_name=collection)
    return create_retriever_tool(
        retriever=vector_db.as_retriever(), name=name, description=description, **kwargs
    )
