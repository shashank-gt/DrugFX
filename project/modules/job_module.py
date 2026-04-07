import json
from llm import generate_response
from rag.retriever import retrieve_context

def analyze_job(job_input: str) -> dict:
    """
    Analyzes a job title or description using RAG and NLP/LLM to provide structured output.
    """
    context = retrieve_context("job", job_input)
    
    system_prompt = """You are an HR and career AI assistant. Your job is to analyze the provided job description or title and return a structured JSON response.
Focus on the India context for salary ranges unless specified otherwise in the text.
Extract key entities like skills and roles specifically.
Format to return EXACTLY:
{
  "required_skills": ["skill1", "skill2"],
  "responsibilities": ["resp1", "resp2"],
  "salary_range": "salary string (India context)",
  "career_growth": "growth string element",
  "work_life_balance": "balance string"
}"""

    prompt = f"Analyze the following job title or description: '{job_input}'\n\n{context}"
    
    response_str = generate_response(prompt=prompt, system_prompt=system_prompt, response_format="json_object")
    
    try:
        return json.loads(response_str)
    except json.JSONDecodeError:
        return {"error": "Failed to parse LLM response into JSON.", "raw_response": response_str}
