from genai_factory.chains.base import ChainRunner
from genai_factory.config import WorkflowServerConfig, get_llm
from genai_factory.tools import tool_registry
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant"


class Agent(ChainRunner):
    def __init__(
        self,
        tools: list[dict] = None,
        llm: BaseChatModel = None,
        system_prompt: str = "",
        verbose: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._tools = tools
        self._llm = llm
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._verbose = verbose
        self._agent = None

    def post_init(self, mode: str = "sync"):
        self._llm = self._llm or get_llm(self.context._config)

        tools = []
        for name, kwargs in self._tools.items():
            try:
                get_tool_fn = tool_registry[name]
                # inject context into all tools
                tool = get_tool_fn(**kwargs, config=self.context._config)
                tools.append(tool)
            except KeyError:
                raise ValueError(
                    f"Tool '{name}' not supported. Expected one of {list(tool_registry.keys())}"
                )
        self._tools = tools

        if self._verbose:
            print(f"Tools: {self._tools}")
            print(f"System Prompt: {self._system_prompt}")

        agent = create_tool_calling_agent(
            llm=self._llm,
            tools=self._tools,
            prompt=ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        self._system_prompt,
                    ),
                    ("user", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            ),
        )
        self._agent = AgentExecutor(
            agent=agent, tools=self._tools, verbose=self._verbose
        )

    def _run(self, event):
        response = list(self._agent.stream({"input": event.query}))
        answer = response[-1]["messages"][-1].content
        return {"answer": answer, "sources": ""}
