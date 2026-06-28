import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

def analyze_user_prompt(prompt: str) -> dict:
    """
    Analyzes the user prompt and extracts company name, start date, and end date using
    JsonOutputParser.
    Returns a dictionary of parameters. If an error occurs, returns an empty dictionary.
    """
    try:
        # Load environment variables
        load_dotenv()
        
        # Calculate date ranges for defaults
        today = datetime.now()
        today_str = today.strftime("%Y%m%d")

        # Initialize ChatOpenAI configured for BigModel
        llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            temperature=0.0
        )
        
        # Initialize JsonOutputParser
        parser = JsonOutputParser()
        
        # Decouple prompt template into components
        core_instruction = (
            "You are a specialized assistant that extracts key research parameters from a user prompt "
            "and formulates a prompt to apply on each individual collected company filing on the Hong Kong Exchange. "
            "The generated prompt must enforce that the analysis relies strictly on ground truths in the filings and "
            "is fully traceable back to specific sources/sections within the filings."
        )
        
        output_format_instruction = (
            "Return a JSON object with the following fields:\n"
            "- \"company_name\": Name of the specific company to research.\n"
            "- \"start_date\": The start date of research in YYYYMMDD format.\n"
            "- \"end_date\": The end date of research in YYYYMMDD format. Default to {today_str} if not specified.\n"
            "- \"filing_analyze_prompt\": A formulated prompt targeting the user's specific questions or criteria to be applied when analyzing each individual collected company filing. This formulated prompt MUST explicitly instruct to strictly rely on ground truths in the filings (avoiding speculation or extrapolation) and trace all analyzed points back to specific sources (e.g., page numbers or section names) within the filings.\n\n"
            "Ensure the output is valid JSON."
        )
        
        input_information = (
            "Today's date is {today_str}.\n"
            "User prompt: {user_prompt}"
        )
        
        # Construct the prompt template
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
        
        # Chain them together
        chain = prompt_template | llm | parser
        
        # Invoke the chain
        result = chain.invoke({
            "today_str": today_str,
            "user_prompt": prompt
        })
        
        return result
            
    except Exception as e:
        print(f"Error during prompt analysis: {e}", file=sys.stderr)
        return {}
