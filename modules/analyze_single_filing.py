import os
from markitdown import MarkItDown
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def analyze_single_filing(
    filing: dict,
    filings_dir: str,
    analysis_prompt: str,
    model: str,
    api_key: str,
    base_url: str,
    idx: int,
    total: int
) -> None:
    """
    Analyzes a single HKEX filing: converts PDF to text using MarkItDown,
    submits it to the LLM with the provided prompt, and writes the summary
    to a file.
    """
    filing_id = filing.get("id")
    filing_name = filing.get("name", "Unknown Name")
    filing_date = filing.get("filing_date", "Unknown Date")
    download_link = filing.get("download_link", "")

    if not filing_id:
        return

    # Check format and skip non-PDF filings
    parsed_url = download_link.split("?")[0].lower()
    if not parsed_url.endswith(".pdf"):
        print(f"[{idx}/{total}] Skipping non-PDF filing '{filing_id}' (Format: {os.path.splitext(parsed_url)[1] or 'Unknown'}).")
        return

    # Verify local PDF file exists
    pdf_filename = f"{filing_id}.pdf"
    pdf_path = os.path.join(filings_dir, pdf_filename)
    analysis_filename = f"{filing_id}-analysis.txt"
    analysis_path = os.path.join(filings_dir, analysis_filename)

    # Check if already analyzed
    if os.path.exists(analysis_path) and os.path.getsize(analysis_path) > 0:
        print(f"[{idx}/{total}] Analysis file '{analysis_filename}' already exists. Skipping analysis (bypass).")
        return

    if not os.path.exists(pdf_path):
        print(f"[{idx}/{total}] PDF file for '{filing_id}' does not exist at '{pdf_path}'. Skipping.")
        return

    print(f"[{idx}/{total}] Starting text extraction and analysis for '{pdf_filename}'...")

    # Initialize local converters to ensure thread safety
    try:
        md_converter = MarkItDown()
        conversion_res = md_converter.convert(pdf_path)
        raw_text = conversion_res.text_content
    except Exception as e:
        print(f"[{idx}/{total}] Error extracting text from '{pdf_filename}': {e}. Skipping.")
        return

    if not raw_text or not raw_text.strip():
        print(f"[{idx}/{total}] Warning: No text extracted from '{pdf_filename}'. Skipping.")
        return

    from modules.document_processor import split_text_semantically, run_stateful_map_reduce

    # Split text into semantic chunks under 100,000 characters
    chunks = split_text_semantically(raw_text, max_chars=100000)
    
    # Process sequentially using stateful map-reduce
    try:
        summary = run_stateful_map_reduce(
            chunks=chunks,
            filing_name=filing_name,
            filing_date=filing_date,
            analysis_prompt=analysis_prompt,
            model=model,
            api_key=api_key,
            base_url=base_url,
            idx=idx,
            total=total,
            pdf_filename=pdf_filename
        )
    except Exception as e:
        print(f"[{idx}/{total}] Error during stateful map-reduce for '{pdf_filename}': {e}. Skipping.")
        return

    # Write output file
    try:
        with open(analysis_path, "w", encoding="utf-8") as out_file:
            out_file.write(f"Filing Name: {filing_name}\n")
            out_file.write(f"Filing Date: {filing_date}\n")
            out_file.write("-" * 50 + "\n")
            out_file.write(summary)
        print(f"[{idx}/{total}] Finished. Analysis written to {analysis_filename}.")
    except Exception as e:
        print(f"[{idx}/{total}] Error writing analysis to file '{analysis_path}': {e}")
