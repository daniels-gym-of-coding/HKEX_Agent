import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_tavily import TavilySearch

def get_hkex_ticker(company_name: str) -> str:
    """
    Retrieves the 5-digit HKEX ticker for a given company name.
    
    Uses Tavily Search to search for the ticker, then queries the LLM
    to extract and format it.
    
    Args:
        company_name: The name of the company to search for.
        
    Returns:
        A 5-digit string representing the HKEX ticker (e.g., "00700").
        
    Raises:
        ValueError: If the ticker is not found, cannot be verified,
                    or the configuration is invalid.
    """
    # Load environment variables
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    load_dotenv(dotenv_path)

    model = os.getenv("LLM_MODEL")
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    tavily_key = os.getenv("TAVILY_API_KEY")

    if not model or not api_key:
        raise ValueError("LLM configuration is missing in the environment.")
    if not tavily_key:
        raise ValueError("Tavily API key is missing in the environment.")

    # 1. Execute Tavily Search
    search = TavilySearch(max_results=3)
    query = f"{company_name} 5-digit HKEX stock ticker code"
    try:
        search_results = search.invoke({"query": query})
    except Exception as e:
        raise ValueError(f"Failed to query Tavily search: {e}")

    # Format the search results into context
    context_items = []
    if isinstance(search_results, list):
        for idx, result in enumerate(search_results):
            title = result.get("title", "No Title")
            content = result.get("content", "")
            url = result.get("url", "")
            context_items.append(f"Result {idx+1}:\nTitle: {title}\nURL: {url}\nContent: {content}\n")
    else:
        context_items.append(str(search_results))
    search_context = "\n".join(context_items)

    # 2. Query the LLM to extract ticker
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.0
    )

    parser = JsonOutputParser()

    # Decouple prompt into several parts as required by rules
    core_instruction = (
        "You are a professional financial assistant specializing in the Hong Kong Stock Exchange (HKEX).\n"
        "Your task is to identify the 5-digit HKEX ticker code for the given company name using the provided search context.\n"
        "Note that HKEX stock tickers are exactly 5 digits (e.g., Tencent is 00700, Alibaba is 09988, Meituan is 03690).\n"
        "If a ticker is shorter than 5 digits (e.g., 5, 700, 9988), pad it with leading zeros to make it 5 digits (e.g., 00005, 00700, 09988).\n"
        "If the correct 5-digit HKEX ticker is not found in the search context or the company is not listed on HKEX, set \"found\" to false."
    )

    output_format_instruction = (
        "Return a JSON object with the following fields:\n"
        "- \"ticker\": A 5-digit string representing the HKEX ticker (e.g., \"00700\"). Set to null if the ticker is not found.\n"
        "- \"found\": A boolean indicating whether the correct 5-digit HKEX ticker was successfully found.\n"
        "- \"reason\": A brief explanation of the result based on the search context.\n\n"
        "Ensure the output is valid JSON."
    )

    input_information = (
        "Company Name: {company_name}\n"
        "Search Context:\n{search_context}"
    )

    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            f"[CORE INSTRUCTION]\n{core_instruction}\n\n"
            f"[OUTPUT FORMAT INSTRUCTION]\n{output_format_instruction}"
        ),
        (
            "user",
            f"[INPUT INFORMATION]\n{input_information}"
        )
    ])

    chain = prompt_template | llm | parser

    try:
        result = chain.invoke({
            "company_name": company_name,
            "search_context": search_context
        })
    except Exception as e:
        raise ValueError(f"Failed to invoke LLM chain to extract ticker: {e}")

    # Validate output
    if not isinstance(result, dict):
        raise ValueError(f"LLM returned an invalid response format: {result}")

    found = result.get("found", False)
    ticker = result.get("ticker")

    if not found or not ticker:
        reason = result.get("reason", "No reason provided.")
        raise ValueError(f"HKEX ticker for company '{company_name}' was not found. Reason: {reason}")

    # Standardize and validate ticker format
    ticker_str = str(ticker).strip()
    if len(ticker_str) < 5 and ticker_str.isdigit():
        ticker_str = ticker_str.zfill(5)

    if len(ticker_str) != 5 or not ticker_str.isdigit():
        raise ValueError(
            f"Retrieved ticker '{ticker_str}' is not a valid 5-digit HKEX ticker. Response: {result}"
        )

    return ticker_str
