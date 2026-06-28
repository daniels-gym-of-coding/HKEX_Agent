import os
from concurrent.futures import ThreadPoolExecutor
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

def run_parallel_map_reduce(
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
    Processes document chunks in parallel (Map phase), then combines the
    results into a single consolidated analysis report (Reduce phase).
    """
    num_chunks = len(chunks)
    if num_chunks == 0:
        return "No text content found to analyze."
    
    print(f"[{idx}/{total}] Map Phase: Processing {num_chunks} chunks in parallel for '{pdf_filename}'...")
    
    # Decouple prompts into core_instruction, output_format_instruction, input_information
    map_core_instruction = (
        "You are an expert financial analyst. Your task is to analyze the text content of a segment (chunk) "
        "of a Hong Kong Exchange filing and extract information matching the user's specific analysis request.\n"
        "Rely strictly on ground truths in the text. Do not extrapolate, speculate, or make assumptions."
    )
    map_output_format = (
        "Provide your analysis summary clearly formatted in Markdown.\n"
        "Focus only on findings relevant to the user's request. If the chunk does not contain relevant information, "
        "simply state that no relevant information was found in this section."
    )
    map_input_info = (
        "Filing Name: {filing_name}\n"
        "Filing Date: {filing_date}\n\n"
        "[FILING TEXT SEGMENT]\n{chunk_text}\n\n"
        "User Analysis Request:\n{analysis_prompt}"
    )
    
    map_prompt_template = ChatPromptTemplate.from_messages([
        ("system", f"[CORE INSTRUCTION]\n{map_core_instruction}\n\n[OUTPUT FORMAT INSTRUCTION]\n{map_output_format}"),
        ("user", f"[INPUT INFORMATION]\n{map_input_info}")
    ])
    
    reduce_core_instruction = (
        "You are an expert financial analyst. Your task is to review multiple individual chunk analyses "
        "from a Hong Kong Exchange filing and consolidate them into a single, cohesive, comprehensive, "
        "and well-structured final report that answers the user's analysis request.\n"
        "Ensure no duplicated findings, organize findings logically, and maintain strict fidelity to the raw data."
    )
    reduce_output_format = (
        "Provide a single, consolidated analysis report in Markdown. "
        "Organize it with clear headings, bullet points, and references to sections/topics as appropriate."
    )
    reduce_input_info = (
        "Filing Name: {filing_name}\n"
        "Filing Date: {filing_date}\n\n"
        "[INDIVIDUAL CHUNK ANALYSES]\n{chunk_analyses}\n\n"
        "User Analysis Request:\n{analysis_prompt}"
    )
    
    reduce_prompt_template = ChatPromptTemplate.from_messages([
        ("system", f"[CORE INSTRUCTION]\n{reduce_core_instruction}\n\n[OUTPUT FORMAT INSTRUCTION]\n{reduce_output_format}"),
        ("user", f"[INPUT INFORMATION]\n{reduce_input_info}")
    ])
    
    def map_chunk(chunk_idx: int, chunk_text: str) -> str:
        return invoke_llm_with_fallback(
            prompt_template=map_prompt_template,
            input_vars={
                "filing_name": filing_name,
                "filing_date": filing_date,
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
        
    map_results = [None] * num_chunks
    with ThreadPoolExecutor(max_workers=min(num_chunks, 8)) as executor:
        futures = {
            executor.submit(map_chunk, chunk_idx, chunk_text): chunk_idx - 1
            for chunk_idx, chunk_text in enumerate(chunks, 1)
        }
        for future in futures:
            chunk_pos = futures[future]
            try:
                map_results[chunk_pos] = future.result()
            except Exception as e:
                print(f"[{idx}/{total}] Error mapping chunk {chunk_pos+1} for '{pdf_filename}': {e}")
                map_results[chunk_pos] = f"Error: Failed to analyze Chunk {chunk_pos+1}."
                
    print(f"[{idx}/{total}] Reduce Phase: Consolidating analyses for '{pdf_filename}'...")
    
    formatted_analyses = ""
    for i, res in enumerate(map_results, 1):
        formatted_analyses += f"--- Chunk {i} Analysis ---\n{res}\n\n"
        
    try:
        final_summary = invoke_llm_with_fallback(
            prompt_template=reduce_prompt_template,
            input_vars={
                "filing_name": filing_name,
                "filing_date": filing_date,
                "chunk_analyses": formatted_analyses,
                "analysis_prompt": analysis_prompt
            },
            model=model,
            api_key=api_key,
            base_url=base_url,
            idx=idx,
            total=total,
            pdf_filename=pdf_filename
        )
        return final_summary
    except Exception as e:
        print(f"[{idx}/{total}] Error during Reduce phase for '{pdf_filename}': {e}")
        raise e
