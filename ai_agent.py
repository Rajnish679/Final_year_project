"""# ai_agent.py

import os
from dotenv import load_dotenv
load_dotenv()

# Load API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent

# System prompt
system_prompt = "Act as an AI chatbot who is smart and friendly"

# Create LLM instances with proper keys
openai_llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)
groq_llm = ChatGroq(model="llama3-70b-8192", groq_api_key=GROQ_API_KEY)

# Define the function to get response
def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    if provider == "Groq":
        llm = ChatGroq(model=llm_id, groq_api_key=GROQ_API_KEY)
    elif provider == "OpenAI":
        llm = ChatOpenAI(model=llm_id, openai_api_key=OPENAI_API_KEY)
    else:
        raise ValueError("Invalid provider. Choose 'Groq' or 'OpenAI'.")

    # Select tools conditionally
    tools = [TavilySearchResults(tavily_api_key=TAVILY_API_KEY, max_results=2)] if allow_search else []

    # Create agent
    agent = create_react_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )

    # Run agent and get result
    state = {"messages": [{"content": query, "role": "user"}]}
    response = agent.invoke(state)

    # Extract AI messages
    messages = response.get("messages", [])
    ai_messages = [m["content"] for m in messages if m.get("role") == "assistant"]

    return ai_messages[-1] if ai_messages else "No response received from AI."

"""









































#old ai_agent code 


# ai_agent.py

import os
from dotenv import load_dotenv
load_dotenv()

# API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# LLMs
openai_llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)
groq_llm = ChatGroq(model="llama3-70b-8192", groq_api_key=GROQ_API_KEY)

# Tool
search_tool = TavilySearchResults(tavily_api_key=TAVILY_API_KEY, max_results=2)

# Agent
agent = create_react_agent(
    model=groq_llm,
    tools=[search_tool]
)
# Dynamic user input
user_query = input("Ask something: ")

# Query with system prompt
state = {
    "messages": [
        SystemMessage(content="You are expert in stocks marketing in India and you have 20+ years of experience in stock buying and selling. Act as my personal guide or adviser help me to make more profit[current time] in Indian stocks market. [ EVERYTIME WHEN YOU GET ANT INOUT DEEP RESEARCH ON WEB AND FIND ALL BEST STOCKS BASED ON RESEARCHED DATA AND SUGGEST ME TO BUY OR SELL IN CURRENT DATE TO MAKE MORE PROFIT]. "),
        HumanMessage(content=user_query)
       
    ]
}

# Agent response
response = agent.invoke(state)

# Extract and print AI message
messages=response.get("messages",[])
ai_messages=[message.content for message in messages if isinstance(message,AIMessage)]

# Print last AI message
if ai_messages:
    print("\nAI:", ai_messages[-1])
else:
    print("No AI response found.")

