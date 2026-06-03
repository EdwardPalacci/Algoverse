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
        "response_format": {"type": "json_object"}, # This forces JSON mode!
        "temperature": 0.7
    }
    
    # Simple logic to save data if successful, or catch errors if it fails
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(OPENROUTER_URL, headers=headers, json=payload)
            return response.json()
        except Exception as e:
            # If primary fails, fallback to backup
            payload["model"] = BACKUP_MODEL
            response = await client.post(OPENROUTER_URL, headers=headers, json=payload)
            return response.json()
