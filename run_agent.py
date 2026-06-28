import os
import sys
import json
from dotenv import load_dotenv


from modules.read_research_topic import read_research_topic
from modules.analyze_user_prompt import analyze_user_prompt
from modules.get_hkex_ticker import get_hkex_ticker
from modules.get_hkex_stock_id import get_hkex_stock_id
from modules.get_hkex_filings import get_hkex_filings
from modules.download_hkex_filings import download_hkex_filings
from modules.analyze_hkex_filings import analyze_hkex_filings
from modules.summarize_analyses import summarize_analyses
from modules.reflect_on_summary import reflect_on_summary


# Load environment variables from .env
load_dotenv()


# Read the research topic prompt
file_path = "research_topic.txt"
prompt = read_research_topic(file_path)

print(f"Analyzing prompt from {file_path}...")

parsed = analyze_user_prompt(prompt)

print(f"Retrieving HKEX ticker for {parsed.get('company_name')}...")
parsed["ticker"] = get_hkex_ticker(parsed.get("company_name"))

print(f"Retrieving stock ID for ticker {parsed['ticker']}...")
parsed["stock_id"] = get_hkex_stock_id(parsed["ticker"])

print("\n--- Parsed Parameters ---")
print(json.dumps(parsed, indent=2))

# Retrieve filings and save to CSV
print(f"\nRetrieving filings for stock ID {parsed['stock_id']} from {parsed.get('start_date')} to {parsed.get('end_date')}...")
filings = get_hkex_filings(
    stock_id=parsed["stock_id"],
    start_date=parsed.get("start_date"),
    end_date=parsed.get("end_date")
)

print(f"Retrieved {len(filings)} filings and saved to filings/filings.csv")


# Download the filings files
print("\nDownloading filing files...")
download_hkex_filings(csv_path="filings/filings.csv")

# Analyze filings
print("\nRunning filing analysis...")
analyze_hkex_filings(parsed_params=parsed, csv_path="filings/filings.csv")

# Summarize analyses (draft)
print("\nRunning consolidated summarization (draft)...")
summarize_analyses(
    company_name=parsed.get("company_name", "the company"),
    filings_dir="filings",
    output_filename="draft_summary.md"
)

# Reflect on summary
print("\nRunning reflection and generating final summary...")
reflect_on_summary(
    company_name=parsed.get("company_name", "the company"),
    draft_summary_path="draft_summary.md",
    filings_dir="filings",
    output_filename="final_summary.md"
)





