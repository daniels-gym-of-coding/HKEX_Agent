import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()

# Read OpenAI API configuration
model = os.getenv("LLM_MODEL")
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

if not api_key or api_key == "your_openai_api_key_here":
    print("Error: No valid LLM configuration found in .env.")
    exit(1)



print(f"Connecting to model/endpoint: {model}...")
print(f"Base URL: {base_url}")

# Initialize ChatOpenAI client
llm = ChatOpenAI(
    model=model,
    api_key=api_key,
    base_url=base_url,
)

# Decouple the prompt into several parts as required by rules:
# - core_instruction
# - output_format_instruction
# - input_information
core_instruction = "Verify that the API key and connection are working properly."
output_format_instruction = "Respond with your model name or identity and a confirmation message."
input_information = "Hi! Tell me your name and if you can hear me."

full_prompt = (
    f"Core Instruction:\n{core_instruction}\n\n"
    f"Output Format Instruction:\n{output_format_instruction}\n\n"
    f"Input Information:\n{input_information}"
)

try:
    response = llm.invoke(full_prompt)
    print("\n--- Connection Success ---")
    print("Response:")
    print(response.content)
except Exception as e:
    print("\n--- Connection Failed ---")
    print("Error:", e)

