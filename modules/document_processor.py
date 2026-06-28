import os
from langchain_core.prompts import ChatPromptTemplate
from modules.llm_client import invoke_llm_with_fallback

def split_text_semantically(text: str, max_chars: int = 100000) -> list[str]:
    """
    Splits markdown/text into chunks semantically. It splits by markdown headers or double newlines,
    ensuring each chunk is under max_chars while keeping sections intact.
    """
    if not text:
        return []
    
    blocks = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_length = 0
    
    for block in blocks:
        block_len = len(block) + 2  # plus separator
        if block_len > max_chars:
            # Flush current chunk
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Split giant block by lines
            lines = block.split("\n")
            sub_chunk = []
            sub_length = 0
            for line in lines:
                line_len = len(line) + 1
                if sub_length + line_len > max_chars:
                    if sub_chunk:
                        chunks.append("\n".join(sub_chunk))
                    sub_chunk = [line]
                    sub_length = line_len
                else:
                    sub_chunk.append(line)
                    sub_length += line_len
            if sub_chunk:
                chunks.append("\n".join(sub_chunk))
            continue
            
        if current_length + block_len > max_chars:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [block]
            current_length = block_len
        else:
            current_chunk.append(block)
            current_length += block_len
            
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
        
    return chunks

def run_stateful_map_reduce(
    chunks: list[str],
    filing_name: str,
    filing_date: str,
    analysis_prompt: str,
    model: str,
    api_key: str,
    base_url: str,
    idx: int,
    total: int,
    pdf_filename: str
) -> str:
    """
    Processes document chunks sequentially, updating a running analysis state
    with failsafe support on each step.
    """
    num_chunks = len(chunks)
    previous_state = "No previous analysis state. This is the first chunk of the document."
    
    # Decouple prompt into several parts as required by rules
    core_instruction = (
        "You are an expert financial analyst. Your task is to analyze the text content of a Hong Kong Exchange filing "
        "and execute the user's specific analysis requirements.\n"
        "You are processing the document in sequential chunks. You must update and refine the running analysis report "
        "based on the new chunk of text while retaining and integrating all previous analysis findings. "
        "Make sure to avoid speculation or extrapolation, relying strictly on ground truths in the text."
    )

    output_format_instruction = (
        "Provide your updated, consolidated analysis summary clearly formatted in Markdown.\n"
        "Ensure all findings from the previous summary state and the new text are integrated. "
        "Maintain professional structure and clear referencing of sources/sections from the text."
    )

    input_information = (
        "Filing Name: {filing_name}\n"
        "Filing Date: {filing_date}\n\n"
        "[PREVIOUS RUNNING ANALYSIS STATE]\n{previous_state}\n\n"
        "[NEW FILING TEXT CHUNK]\n{chunk_text}\n\n"
        "User Analysis Request:\n{analysis_prompt}"
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

    for chunk_idx, chunk_text in enumerate(chunks, 1):
        print(f"[{idx}/{total}] Processing chunk {chunk_idx}/{num_chunks} for '{pdf_filename}'...")
        
        # Invoke the failsafe LLM caller
        previous_state = invoke_llm_with_fallback(
            prompt_template=prompt_template,
            input_vars={
                "filing_name": filing_name,
                "filing_date": filing_date,
                "previous_state": previous_state,
                "chunk_text": chunk_text,
                "analysis_prompt": analysis_prompt
            },
            model=model,
            api_key=api_key,
            base_url=base_url,
            idx=idx,
            total=total,
            pdf_filename=f"{pdf_filename} (Chunk {chunk_idx}/{num_chunks})"
        )
        
    return previous_state
