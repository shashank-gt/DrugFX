import os
import json
from typing import Optional

try:
    from openai import OpenAI
    HAS_OPENAI_LIB = True
except ImportError:
    HAS_OPENAI_LIB = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Initialize OpenAI client if API key is available
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if HAS_OPENAI_LIB and api_key else None

def generate_response(prompt: str, system_prompt: str = "You are a helpful AI assistant.", model: str = "gpt-3.5-turbo", response_format: str = "text") -> str:
    """
    Sends a prompt to the LLM and returns the response.
    Includes a mock fallback if no API key or openai library is provided.
    """
    if not client:
        print("WARNING: OPENAI_API_KEY not found or openai library missing. Using mock LLM response.")
        if "drug" in system_prompt.lower() or "drug" in prompt.lower():
            return json.dumps({
                "uses": ["Pain relief", "Fever reduction"],
                "side_effects": ["Nausea", "Dizziness"],
                "dosage": "500mg every 4-6 hours",
                "warnings": ["Consult a medical professional before use", "Do not take with alcohol"],
                "drug_interactions": ["Blood thinners"],
                "alternatives": ["Ibuprofen", "Naproxen"]
            }, indent=2)
        elif "job" in system_prompt.lower() or "job" in prompt.lower():
            return json.dumps({
                "required_skills": ["Python", "Machine Learning", "Communication"],
                "responsibilities": ["Develop AI models", "Deploy to production"],
                "salary_range": "\u20b910,00,000 - \u20b925,00,000",
                "career_growth": "AI Engineer -> Lead AI Engineer -> AI Architect",
                "work_life_balance": "Generally good, remote options available"
            }, indent=2)
        return json.dumps({"result": "Mock LLM Response"})

    try:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        }
        if response_format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return "{}"
