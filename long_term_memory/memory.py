import argparse
import os
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import pinecone
import tiktoken
from langchain.agents import load_tools
from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.chat_models.openai import ChatOpenAI
from langchain.embeddings.base import Embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.tools import Tool
from langchain.utilities import SerpAPIWrapper
from langchain.vectorstores import Pinecone
from tiktoken import Encoding

SYSTEM_MESSAGE = SystemMessage(
    content="You are an insightful, kind, helpful, detail oriented AI assistant who has creative solutions that are explained in detail."
)
PINECONE_ENV = "gcp-starter"
SEPERATORS = ["\n\n", "\n", " ", ""]
EMBEDDING_MODEL = "text-embedding-ada-002"

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
SERPAPI_API_KEY = os.environ["SERPAPI_KEY"]


def create_conversational_retrieval_agent(
    llm: ChatOpenAI,
    tools: List[Tool],
    memory_key: str = "chat_history",
    system_message: Optional[SystemMessage] = None,
    verbose: bool = False,
    max_token_limit: int = 2000,
    **kwargs: "Any",
) -> AgentExecutor:
    """Create a conversational retrival agent.

    OpenAI's conversational retrival agent.

    Args:
        llm: An OpenAI.
        tools: A list of tools that the LLM can use.
        memory_key: The key that will be injected into the prompts.
            Defaults to "chat_history".
        system_message: First message to set the context of the
            conversation for the LLM agent. Defaults to None.
        verbose: Whether or not to print the verbose trace of the
            agent traction. Defaults to False.
        max_token_limit: Max number of tokens to keep around in memory.
            Defaults to 2000.

    Returns:
        An agent executor.
    """
    # NOTE: This memory chain saves the intermediate steps the agent takes as well as the output.
    memory = AgentTokenBufferMemory(
        memory_key=memory_key,
        llm=llm,
        max_token_limit=max_token_limit,
    )

    # NOTE: This type of prompt template thinks about a list of BaseMessages
    # either from the agent, human, or system itself (system is like the context)
    # The way to think about
    # ChatPromptTemplate defines a set of input variables (like user_input or chat history)
    # that will generate the prompt that's fed to the user.
    # What's happening below under the hood is we're creating something like
    #     template = ChatPromptTemplate(messages=[
    #     ("system", system_message),
    #     ("system", extra_prompt_messages[0])
    #     ...
    #     ("human", "{input}"),
    #     ("ai", "{agent_scratch_pad}")
    # ])
    # Ultimatly this will create a string representation of the input that's fed to the agent model
    # and is updated as the agent takes actions
    prompt: ChatPromptTemplate = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)],
    )

    # NOTE: The llm that's taking actions, the tools that it has access to and can select from,
    # and the prompt formatting.
    agent = OpenAIFunctionsAgent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=verbose,
        return_intermediate_steps=True,
        **kwargs,
    )


def pinecone_tool(
    index: str,
    text_splitter: TextSplitter,
    embedder: Embeddings,
    dimension: int,
    text_key: str = "text",
) -> Tuple[Pinecone, Tool, Tool]:
    """Create a Pinecone Vector database query tool.

    Args:
        index: Name of the index create in Pinecone.
        text_splitter: How to split up the text.
        embedder: An embedder object that takes as input
            a string and outputs a vector of dimension d.
        dimension: Dimension size.
        text_key: Metadata key for storing the embedding. Defaults to "text".

    Returns:
        Tuple[Pinecone database, A retriver tool, A storing tool]
    """
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV,
    )

    if index not in pinecone.list_indexes():
        pinecone.create_index(
            index,
            metric="cosine",
            dimension=dimension,
        )

    vector_store = Pinecone(
        index=pinecone.Index(
            index_name=index,
            pool_threads=8,
        ),
        embedding=embedder.embed_query,
        text_key=text_key,
    )

    search_tool = create_retriever_tool(
        retriever=vector_store.as_retriever(
            search_kwargs={
                "k": 3,
            },
            return_source_documents=True,
        ),
        name="long_term_memory",
        description="AI's long term memory about previous conversations.",
    )

    # pylint: disable-next: missing-function-docstring, missing-return-doc
    def _store_tool(text: str) -> str:
        grpc_index = pinecone.GRPCIndex(index)
        record_texts: List[str] = text_splitter.split_text(text)
        _record_metadata: List[Dict[str, Any]] = [
            {"chunk": j, "text": text} for j, text in enumerate(record_texts)
        ]
        ids = [str(uuid4()) for _ in range(len(record_texts))]
        embeds = embedder.embed_documents(record_texts)
        grpc_index.upsert(
            vectors=zip(
                ids,
                embeds,
                _record_metadata,
            )
        )
        return text

    store_tool = Tool.from_function(
        func=_store_tool,
        name="store_memory",
        description="AI's storage long term memory of current conversation but only when asked by the user",
    )

    return (
        vector_store,
        search_tool,
        store_tool,
    )


def search_api_tool(
    engine: str = "google",  # It's missing input validation but for now this is fine.
    country_code: str = "gl",
    host_language: str = "en",
) -> Tool:
    """Contruct a Serp API search langchain tool.

    Requires an API key.

    Args:
        engine: Engine name. Defaults to "google",
        host_language: Host language. Defaults to "en".

    Returns:
        An agent tool.
    """
    serpapi = SerpAPIWrapper(
        serpapi_api_key=SERPAPI_API_KEY,
        params={
            "engine": engine,
            "gl": country_code,
            "hl": host_language,
        },
    )
    return Tool.from_function(
        func=serpapi.run,
        name="search",
        description="A useful tool for searching the internet",
    )


def run(
    query: Optional[str] = None,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
    dimension: int = 1536,
    model_name: str = "gpt-3.5-turbo",
    temperature: float = 0.5,
    loop_with_agent: bool = False,
    verbose: bool = True,
) -> None:
    """Logic for initalizing and running a langchain agent with various tools.

    Conversational LangChain agent powered by ChatOpenAI, Pinecone, SerpAPI.

    Args:
        query: User query for agent. Defaults to None.
        chunk_size: Text size for embedding. Defaults to 400.
        chunk_overlap: How much overlap b/w chunks when
            text is split before embedding. Defaults to 50.
        dimension: Embedder dimension. Defaults to 1536.
        model_name: OpenAI GPT model name. Defaults to "gpt-3.5-turbo".
        temperature: Model temperature, higher is more variance.
            Defaults to 0.5.
        loop_with_agent: Whether or not we enter into a loop with the agent.
            Defaults to False.
        verbose: Whether or not we want to show the agent trace.
            Defaults to True.

    """
    tokenizer = tiktoken.get_encoding("cl100k_base")

    def find_tokens(tokenizer: Encoding, text: str) -> Tuple[List[int], int]:
        tokens: List[int] = tokenizer.encode(text, disallowed_special=())
        return tokens, len(tokens)

    tokenizer = tiktoken.get_encoding("cl100k_base")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda text: find_tokens(tokenizer, text)[1],
        separators=SEPERATORS,
    )
    embedder = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
        show_progress_bar=True,
    )
    (
        _db_vector_store,
        db_search_tool,
        db_store_tool,
    ) = pinecone_tool(
        index=PINECONE_INDEX_NAME,
        text_splitter=text_splitter,
        embedder=embedder,
        dimension=dimension,
    )

    # NOTE: that this model and agent is not syncronous
    # despite setting streaming to true
    open_ai_llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name=model_name,
        temperature=temperature,
        streaming=True,
    )

    search_tool = search_api_tool()
    other_tools = load_tools(["wikipedia"])
    tools = [
        db_search_tool,
        db_store_tool,
        search_tool,
        *other_tools,
    ]
    convo_openai_agent = create_conversational_retrieval_agent(
        llm=open_ai_llm,
        tools=tools,
        verbose=verbose,
    )

    def ask_question(query: str) -> None:
        print(convo_openai_agent({"input": query})["output"])

    if query is not None:
        ask_question(query)
    if loop_with_agent:
        while True:
            print("Human input: ")
            query = input()
            ask_question(query)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        type=Optional[str],
        help="Single query we want to ask our agent.",
        default=None,
    )
    parser.add_argument(
        "--loop", action="store_true", help="Enable looping with the agent."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose outputs.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        choices=[
            "gpt-3.5-turbo",
            "gpt-4",
        ],
        default="gpt-3.5-turbo",
    )
    args = parser.parse_args()
    run(
        query=args.query,
        loop_with_agent=args.loop,
        verbose=args.verbose,
        model_name=args.model_name,
    )
