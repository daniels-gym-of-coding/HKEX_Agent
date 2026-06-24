import os
import sys
import json
from dotenv import load_dotenv


from modules.read_research_topic import read_research_topic
from modules.analyze_user_prompt import analyze_user_prompt
from modules.get_hkex_ticker import get_hkex_ticker

# Load environment variables from .env
load_dotenv()

# Read the research topic prompt
file_path = "research_topic.txt"
prompt = read_research_topic(file_path)

print(f"Analyzing prompt from {file_path}...")

parsed = analyze_user_prompt(prompt)
parsed["ticker"] = get_hkex_ticker(parsed.get("company_name"))

print("\n--- Success ---")
print("Parsed Parameters:")
print(json.dumps(parsed, indent=2))
