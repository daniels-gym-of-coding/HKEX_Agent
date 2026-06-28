import os
import glob
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def reflect_on_summary(
    company_name: str,
    draft_summary_path: str,
    filings_dir: str = "filings",
    output_filename: str = "final_summary.md"
) -> None:
    """
    Reads the draft summary report, calls the LLM to perform reflection,
    critical review, and self-correction against the original filing analyses,
    and writes the polished final summary report to the same parent directory.
    """
    load_dotenv()
    model = os.getenv("LLM_MODEL")
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not model or not api_key:
        raise ValueError("LLM configuration (LLM_MODEL or OPENAI_API_KEY) is missing from the environment.")

    # Determine absolute path of draft summary
    if not os.path.isabs(draft_summary_path):
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        draft_summary_path = os.path.join(current_dir, draft_summary_path)

    parent_dir = os.path.dirname(draft_summary_path)
    output_path = os.path.join(parent_dir, output_filename)

    if not os.path.exists(draft_summary_path):
        print(f"Draft summary file not found at '{draft_summary_path}'. Reflection aborted.")
        return

    print(f"Reading draft summary from {draft_summary_path}...")
    try:
        with open(draft_summary_path, "r", encoding="utf-8") as f:
            draft_summary_content = f.read().strip()
    except Exception as e:
        print(f"Error reading draft summary file: {e}")
        return

    if not draft_summary_content:
        print("Draft summary is empty. Reflection aborted.")
        return

    # Normalize filings directory path
    if not os.path.isabs(filings_dir):
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filings_dir = os.path.join(current_dir, filings_dir)

    # Find and load the original individual filing analyses (ground truth)
    search_pattern = os.path.join(filings_dir, "*-analysis.txt")
    analysis_files = glob.glob(search_pattern)

    if not analysis_files:
        print(f"Warning: No filing analysis files (*-analysis.txt) found in '{filings_dir}' for reflection. Proceeding without ground truth.")
        analyses_text = "(No source analyses files found)"
    else:
        print(f"Found {len(analysis_files)} analysis files for reflection factuality check.")
        concatenated_parts = []
        for file_path in sorted(analysis_files):
            filename = os.path.basename(file_path)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        concatenated_parts.append(f"### File: {filename}\n{content}\n")
            except Exception as e:
                print(f"Warning: Failed to read {filename} during reflection: {e}")
        analyses_text = "\n" + ("-" * 60 + "\n").join(concatenated_parts)

    print("Submitting draft summary and source analyses for critical reflection and final polishing...")

    # Set up LLM components
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.0
    )

    # Decouple prompt into components as required by project rules
    core_instruction = (
        "You are a critical reviewer and senior financial analyst.\n"
        "Your task is to perform reflection, fact-checking, and self-correction on a draft research summary report.\n"
        "Verify every claim in the draft report against the original Individual Filing Analyses (Ground Truth). Correct any "
        "hallucinations, omissions, or logical inconsistencies, and verify that all source citations match "
        "the files from which the information was extracted. Generate a polished, final summary report that is 100% "
        "accurate to the source material, fully traceable, and audit-ready."
    )

    output_format_instruction = (
        "Provide a polished, final Markdown report. Keep the key structured sections from the draft:\n"
        "1. # Executive Summary Report for {company_name}\n"
        "2. ## Executive Summary\n"
        "3. ## Consolidated Summary Table\n"
        "4. ## Detailed Findings\n"
        "5. ## Conclusion and Recommendations\n\n"
        "Refine the language, correct any logical inconsistencies, ensure risk ratings match the analysis details, "
        "verify that every detailed finding is traced back to a specific file, and format tables and quotes cleanly."
    )

    input_information = (
        "Company Name: {company_name}\n\n"
        "Original Individual Filing Analyses (Ground Truth):\n{analyses_text}\n\n"
        "Draft Summary Content:\n{draft_summary_content}"
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

    chain = prompt_template | llm | StrOutputParser()

    try:
        final_report = chain.invoke({
            "company_name": company_name,
            "draft_summary_content": draft_summary_content,
            "analyses_text": analyses_text
        })
    except Exception as e:
        print(f"Error during reflection summary generation: {e}")
        return

    # Save to the output file
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_report)
        print(f"Polished final summary report successfully written to parent directory: {output_filename}")
    except Exception as e:
        print(f"Error writing final summary report to '{output_path}': {e}")
