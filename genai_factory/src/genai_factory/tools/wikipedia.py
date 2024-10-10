from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


def get_wikipedia_tool(
    top_k_results=1, doc_content_chars_max=100, description: str = None, **kwargs
) -> WikipediaQueryRun:

    default_description = WikipediaQueryRun(api_wrapper="").description

    d = description or default_description

    return WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(
            top_k_results=top_k_results,
            doc_content_chars_max=doc_content_chars_max,
            description=description or default_description,
            **kwargs
        )
    )
