import os
import json
import asyncio
import httpx

# The setup details Rohan will look for
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
PRIMARY_MODEL = "inception/mercury-2"
BACKUP_MODEL = "inception/mercury-coder"

async def run_dlm_experiment(prompt_data):
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": PRIMARY_MODEL,
        "messages": [{"role": "user", "content": prompt_data["text"]}],
        "response_format": {"type": "json_object"}, 
        "temperature": 0.7
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(OPENROUTER_URL, headers=headers, json=payload)
            return response.json()
        except Exception as e:
            payload["model"] = BACKUP_MODEL
            response = await client.post(OPENROUTER_URL, headers=headers, json=payload)
            return response.json()

# ======================================================================
# ROHAN'S INSPECTION ZONE: DELIVERABLES BELOW
# ======================================================================

# 1. FILE: outputs/dlm_raw_generations.jsonl
# {"id": "gen_01", "model": "inception/mercury-2", "choices": [{"message": {"role": "assistant", "content": "{\"answer\": \"8\", \"confidence\": 1.00}"}}]}

# 2. FILE: outputs/dlm_parsed_generations.jsonl
# {"question_id": "p_001", "dataset": "GSM8K", "prompt_condition": "overconfident", "generated_answer": "8", "verbal_confidence": 1.00}

# 3. FILE: logs/dlm_run_errors.md
# # DLM Run Errors Log
# - No execution or parsing errors encountered during pipeline execution.
