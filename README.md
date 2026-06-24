# HKEX_Agent  

## Prerequisites

LLM API Key, Tavily API Key

## Data Source

https://www.hkexnews.hk/index.htm

## Analyze Request

The agent would receive a user prompt stored in <research_topic.txt> to inquire some aspects about one specific company across certain periods.<br><br>
After analyzing the user prompt, the agent should specify key parameters for collecing files with a tool and a specific prompt to apply on collected filings.<br><br>

## Use Tools

A tool would be given to the agent for fetching HKEX stockId with stock ticker first. <br>
A tool would be given to the agent for fetching HKEX filings with stockId, start_date, end_date.<br>

## Analyze Files

for each file, the specific prompt would be used to extract certain information.<br> 

## Summarize, Reflect

The agent then would collect all those extracted information and summarize.<br>
The agent would reflect on the draft summary and generate the final summary.<br>
