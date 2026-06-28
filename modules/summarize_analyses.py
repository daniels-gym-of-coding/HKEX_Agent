import os
import glob
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def summarize_analyses(
    company_name: str,
    filings_dir: str = "filings",
    output_filename: str = "draft_summary.md"
) -> None:
    """
    Reads all individual filing analysis files in the filings directory,
    concatenates their content, and calls the LLM to generate a consolidated
    draft Summary Report in the parent directory.
    """
    load_dotenv()
    model = os.getenv("LLM_MODEL")
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not model or not api_key:
        raise ValueError("LLM configuration (LLM_MODEL or OPENAI_API_KEY) is missing from the environment.")

    # Normalize filings directory path
    if not os.path.isabs(filings_dir):
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filings_dir = os.path.join(current_dir, filings_dir)

    parent_dir = os.path.dirname(filings_dir)
    output_path = os.path.join(parent_dir, output_filename)

    # Find all analysis files (excluding the final summary itself if it exists there)
    search_pattern = os.path.join(filings_dir, "*-analysis.txt")
    analysis_files = glob.glob(search_pattern)

    if not analysis_files:
        print(f"No filing analysis files (*-analysis.txt) found in '{filings_dir}'. Cannot generate summary.")
        return

    print(f"Found {len(analysis_files)} individual analysis files. Concatenating content...")

    # Read and concatenate content
    concatenated_parts = []
    # Sort files by name/id to keep some order
    for file_path in sorted(analysis_files):
        filename = os.path.basename(file_path)
        original_pdf = filename.replace("-analysis.txt", ".pdf")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    concatenated_parts.append(f"### Original Filing: {original_pdf}\n{content}\n")
        except Exception as e:
            print(f"Warning: Failed to read {filename}: {e}")

    if not concatenated_parts:
        print("No valid content found in analysis files. Summary aborted.")
        return

    analyses_text = "\n" + ("-" * 60 + "\n").join(concatenated_parts)

    print("Sending concatenated analyses to LLM for final summary generation...")

    # Set up LLM components
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.0
    )

    core_instruction = (
        "You are a professional research analyst.\n"
        "Your task is to review a series of individual filing analyses for a company and synthesize them into a single, "
        "cohesive, high-level summary report.\n"
        "CRITICAL FACTUALITY RULE: Base your summary strictly and exclusively on the provided Individual Filing Analyses. "
        "Do not assume, speculate, or introduce any information not explicitly present in the input text."
    )

    output_format_instruction = (
        "Provide a professional Markdown report structured as follows:\n"
        "1. # Executive Summary Report for {company_name}\n"
        "2. ## Executive Summary: A high-level assessment of the overall findings and key takeaways based on all filings.\n"
        "3. ## Consolidated Summary Table: A table listing the key topics/criteria analyzed, their consolidated assessments or status, and a brief rationale.\n"
        "4. ## Detailed Findings: Group and synthesize all findings into the key thematic categories or criteria that were analyzed in the individual filings. Under each category, consolidate the findings from all filings. Every point must include a citation to the original filing (e.g., [Filing: 12219063.pdf] or the actual filing name like \"Annual Report 2023\") and its filing date to ensure 100% traceability to the original source filings.\n"
        "5. ## Conclusion and Recommendations: Next steps or key areas that require close monitoring.\n\n"
        "Ensure the output is written in a professional tone in English or Chinese depending on the dominant language of the inputs."
    )

    input_information = (
        "Company Name: {company_name}\n\n"
        "Individual Filing Analyses:\n{analyses_text}"
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
        summary_report = chain.invoke({
            "company_name": company_name,
            "analyses_text": analyses_text
        })
    except Exception as e:
        print(f"Error during final summary generation: {e}")
        return

    # Save to the output file
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(summary_report)
        print(f"Consolidated summary report successfully written to: {output_filename}")
    except Exception as e:
        print(f"Error writing summary report to '{output_path}': {e}")
