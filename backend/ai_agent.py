import os
from dotenv import load_dotenv 
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    if provider == "Groq":
        llm = ChatGroq(model=llm_id, groq_api_key=os.getenv("GROQ_API_KEY"))
    elif provider == "OpenAI":
        llm = ChatOpenAI(model=llm_id, openai_api_key=os.getenv("OPENAI_API_KEY"))
    else:
        raise ValueError("Invalid provider")

    tools = [TavilySearchResults(tavily_api_key=os.getenv("TAVILY_API_KEY"), max_results=2)] if allow_search else []

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt
    )

    state = {"messages": [HumanMessage(content=query)]}
    response = agent.invoke(state)

    messages = response.get("messages", [])
    ai_messages = [msg.content for msg in messages if isinstance(msg, AIMessage)]
    return ai_messages[-1] if ai_messages else "No AI response received."