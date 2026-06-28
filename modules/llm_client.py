import os
import time
import threading
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Thread lock and tracking variable for rate limiting Gemini requests
_gemini_lock = threading.Lock()
_last_gemini_request_time = 0.0

def invoke_llm_with_fallback(
    prompt_template: ChatPromptTemplate,
    input_vars: dict,
    model: str,
    api_key: str,
    base_url: str,
    idx: int,
    total: int,
    pdf_filename: str
) -> str:
    """
    Invokes the primary LLM (e.g. GLM-5.1). If it fails, falls back to Gemini 3.5 Flash
    using ChatOpenAI configured with a custom OpenAI-compatible endpoint.
    Restricts the model to gemini-3.5-flash and caps Gemini usage to at most 1 request
    per minute using a thread-safe lock.

    Args:
        prompt_template: ChatPromptTemplate to run.
        input_vars: Dictionary of variables for the prompt template.
        model: Primary model name.
        api_key: Primary API key.
        base_url: Primary Base URL.
        idx: Current filing index.
        total: Total number of filings.
        pdf_filename: File name of the PDF being analyzed.

    Returns:
        The text response from the LLM.
    """
    # 1. Attempt primary LLM call
    try:
        primary_llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.0
        )
        chain = prompt_template | primary_llm | StrOutputParser()
        return chain.invoke(input_vars)
    except Exception as e:
        print(f"[{idx}/{total}] Error calling primary LLM ({model}) for '{pdf_filename}': {e}.")
        
        # 2. Attempt fallback LLM calls
        gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not gemini_api_key:
            raise ValueError(
                f"[{idx}/{total}] Gemini API key (GEMINI_API_KEY or GOOGLE_API_KEY) is missing from the environment. "
                "Failsafe aborted."
            ) from e
        
        gemini_base_url = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta/openai/")
        fallback_models = ["gemini-3.5-flash", "gemini-3-flash-preview"]
        last_err = None
        
        # Enforce rate limit (at most 1 request per minute) using the global lock
        global _last_gemini_request_time
        with _gemini_lock:
            now = time.time()
            elapsed = now - _last_gemini_request_time
            if elapsed < 60.0:
                sleep_time = 60.0 - elapsed
                print(f"[{idx}/{total}] Rate limit active (1 req/min). Sleeping for {sleep_time:.2f} seconds before calling Gemini...")
                time.sleep(sleep_time)
            # Update the last request time right before invoking the API
            _last_gemini_request_time = time.time()
            
        for fallback_model in fallback_models:
            print(f"[{idx}/{total}] Attempting failsafe: falling back to {fallback_model}...")
            try:
                fallback_llm = ChatOpenAI(
                    model=fallback_model,
                    api_key=gemini_api_key,
                    base_url=gemini_base_url,
                    temperature=0.0
                )
                fallback_chain = prompt_template | fallback_llm | StrOutputParser()
                result = fallback_chain.invoke(input_vars)
                print(f"[{idx}/{total}] Failsafe successful. Analysis generated using {fallback_model}.")
                return result
            except Exception as fallback_err:
                print(f"[{idx}/{total}] {fallback_model} failed: {fallback_err}.")
                last_err = fallback_err
                
        raise ValueError(
            f"[{idx}/{total}] Both primary LLM ({model}) and all fallback LLMs failed. "
            f"Primary error: {e}. Last fallback error: {last_err}"
        ) from last_err
