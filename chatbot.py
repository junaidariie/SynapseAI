
from typing import TypedDict, Annotated
from langchain_core.messages import (
    BaseMessage,
    SystemMessage
)
from langgraph.checkpoint.memory import MemorySaver
from tools import retriever, create_rag_tool, arxiv_search, calculator, get_stock_price, wikipedia_search, tavily_search, convert_currency, unit_converter, get_news, get_joke, get_quote, get_weather
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
import os
load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# =====================================================
# 1️⃣ SYSTEM PROMPT
# =====================================================

SYSTEM_PROMPT = SystemMessage(
    content="""
You are an intelligent AI assistant built by Junaid.

Your role is to provide clear, concise, and human-friendly explanations.

━━━━━━━━━━━━━━━━━━━━━━
🔹 DOCUMENT HANDLING RULES (VERY IMPORTANT)
━━━━━━━━━━━━━━━━━━━━━━
When using retrieved documents:

1. NEVER repeat raw document text verbatim.
2. NEVER list large copied sections from documents.
3. ALWAYS summarize and interpret information in your own words.
4. Organize information logically and clearly.
5. Focus on meaning, not raw content.

If the user asks:
- "What is this document about?"
→ Provide a high-level summary (3–6 sentences).

- "Explain the document"
→ Provide structured explanation with sections.

- "List key points"
→ Provide clean bullet points (max 6).

━━━━━━━━━━━━━━━━━━━━━━
🔹 RAG PRIORITY
━━━━━━━━━━━━━━━━━━━━━━
- Use retrieved content as your *knowledge base*.
- Do NOT hallucinate.
- If the document does not contain the answer, say so clearly.

━━━━━━━━━━━━━━━━━━━━━━
🔹 COMMUNICATION STYLE
━━━━━━━━━━━━━━━━━━━━━━
- Be concise, human, and clear.
- Avoid repetition.
- Avoid technical verbosity unless requested.
- Prefer clarity over completeness.

━━━━━━━━━━━━━━━━━━━━━━
🔹 IDENTITY
━━━━━━━━━━━━━━━━━━━━━━
You are the official AI assistant of Junaid’s AI system.
You help users understand complex information simply and accurately.
"""
)




# =====================================================
# 4️⃣ STATE
# =====================================================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# =====================================================
# 5️⃣ LLM + TOOLS
# =====================================================

llm = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0.4,
    streaming=True
)

rag_tool = create_rag_tool()

tools = [rag_tool, get_stock_price, calculator, wikipedia_search, arxiv_search, tavily_search, convert_currency, unit_converter, get_news, get_joke, get_quote, get_weather]
llm = llm.bind_tools(tools)
tool_node = ToolNode(tools)


# =====================================================
# 6️⃣ CHAT NODE
# =====================================================

def chatbot(state: ChatState):
    messages = [SYSTEM_PROMPT] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}



# =====================================================
# 7️⃣ GRAPH
# =====================================================
memory = MemorySaver()
graph = StateGraph(ChatState)

graph.add_node("chat", chatbot)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat")
graph.add_conditional_edges("chat", tools_condition)
graph.add_edge("tools", "chat")
graph.add_edge("chat", END)

app = graph.compile(checkpointer=memory)