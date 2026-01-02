from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import os
import requests

load_dotenv()

API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# -------------------------------
# GLOBAL RETRIEVER
# -------------------------------
retriever = None


def build_vectorstore(path: str):
    loader = PyPDFLoader(path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    split_docs = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(split_docs, embeddings)


def update_retriever(pdf_path: str):
    global retriever
    vectorstore = build_vectorstore(pdf_path)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# -------------------------------
# RAG TOOL
# -------------------------------
def create_rag_tool():

    @tool
    def rag_search(query: str) -> str:
        """
        Retrieve relevant information from uploaded documents.
        """
        if retriever is None:
            return "No document uploaded yet."

        docs = retriever.invoke(query)

        if not docs:
            return "No relevant information found."

        return "\n\n".join(d.page_content for d in docs)

    return rag_search

@tool
def arxiv_search(query: str) -> dict:
    """
    Search arXiv for academic papers related to the query.
    """
    try:
        arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
        results = arxiv.run(query)
        return {"query": query, "results": results}
    except Exception as e:
        return {"error": str(e)}
    
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}
@tool
def tavily_search(query: str) -> dict:
    """
    Perform a web search using Tavily,
    also use it to get weather information,
    Returns up to 5 search results.
    """
    try:
        search = TavilySearchResults(max_results=5)
        results = search.run(query)
        return {"query": query, "results": results}
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={API_KEY}"
    r = requests.get(url)
    return r.json()

@tool
def wikipedia_search(query: str) -> dict:
    """
    Search Wikipedia for a given query and return results.
    """
    try:
        wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        results = wiki.run(query)
        return {"query": query, "results": results}
    except Exception as e:
        return {"error": str(e)}

@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """
    Convert amount from one currency to another using Frankfurter API.
    Example: convert_currency(100, "USD", "EUR")
    """
    try:
        url = f"https://api.frankfurter.app/latest?amount={amount}&from={from_currency}&to={to_currency}"
        r = requests.get(url)
        return r.json()
    except Exception as e:
        return {"error": str(e)}
@tool


def unit_converter(value: float, from_unit: str, to_unit: str) -> dict:
    """
    Convert between metric/imperial units (supports: km<->miles, kg<->lbs, C<->F).
    Example: unit_converter(10, "km", "miles")
    """
    try:
        conversions = {
            ("km", "miles"): lambda x: x * 0.621371,
            ("miles", "km"): lambda x: x / 0.621371,
            ("kg", "lbs"): lambda x: x * 2.20462,
            ("lbs", "kg"): lambda x: x / 2.20462,
            ("C", "F"): lambda x: (x * 9/5) + 32,
            ("F", "C"): lambda x: (x - 32) * 5/9
        }
        if (from_unit, to_unit) not in conversions:
            return {"error": f"Unsupported conversion: {from_unit} -> {to_unit}"}
        result = conversions[(from_unit, to_unit)](value)
        return {"value": value, "from": from_unit, "to": to_unit, "result": result}
    except Exception as e:
        return {"error": str(e)}



@tool
def get_news(query: str) -> dict:
    """
    Fetch latest news headlines for a given query.
    Example: get_news("artificial intelligence")
    """
    try:
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}&language=en"
        r = requests.get(url)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


@tool
def get_joke(category: str = "Any") -> dict:
    """
    Get a random joke. Categories: Programming, Misc, Pun, Spooky, Christmas, Any
    Example: get_joke("Programming")
    """
    try:
        url = f"https://v2.jokeapi.dev/joke/{category}"
        r = requests.get(url)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

@tool
def get_quote(tag: str = "") -> dict:
    """
    Fetch a random quote. Optionally filter by tag (e.g., 'inspirational', 'technology').
    Example: get_quote("inspirational")
    """
    try:
        url = f"https://api.quotable.io/random"
        if tag:
            url += f"?tags={tag}"
        r = requests.get(url)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

@tool
def get_weather(city: str) -> dict:
    """
    Get current weather for a given city using WeatherAPI.com.
    Example: get_weather("London")
    """
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}&aqi=no"
        r = requests.get(url)
        data = r.json()

        if "error" in data:
            return {"error": data["error"]["message"]}

        return {
            "location": data["location"]["name"],
            "country": data["location"]["country"],
            "temperature_c": data["current"]["temp_c"],
            "temperature_f": data["current"]["temp_f"],
            "condition": data["current"]["condition"]["text"],
            "humidity": data["current"]["humidity"],
            "wind_kph": data["current"]["wind_kph"],
            "wind_dir": data["current"]["wind_dir"]
        }
    except Exception as e:
        return {"error": str(e)}
    


@tool
def get_news(query: str) -> dict:
    """
    Fetch latest news headlines for a given query.
    Example: get_news("artificial intelligence")
    """
    try:
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}&language=en"
        r = requests.get(url)
        return r.json()
    except Exception as e:
        return {"error": str(e)}