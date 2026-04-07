import json
from llm import generate_response
from rag.retriever import retrieve_context

def analyze_drug(drug_input: str) -> dict:
    """
    Analyzes a drug name or description using RAG and LLM to provide structured output.
    """
    context = retrieve_context("drug", drug_input)
    
    system_prompt = """You are a medical AI assistant. Your job is to analyze the provided drug name/description and return a structured JSON response.
Do NOT include any markdown formatting outside of the JSON block if using json_object format.
Always include this exact disclaimer in the warnings: 'Consult a medical professional before use'.
Format to return EXACTLY:
{
  "uses": ["use1", "use2"],
  "side_effects": ["effect1", "effect2"],
  "dosage": "dosage string",
  "warnings": ["warning1", "warning2"],
  "drug_interactions": ["interaction1"],
  "alternatives": ["alt1", "alt2"]
}"""

    prompt = f"Analyze the following drug text: '{drug_input}'\n\n{context}"
    
    response_str = generate_response(prompt=prompt, system_prompt=system_prompt, response_format="json_object")
    
    try:
        return json.loads(response_str)
    except json.JSONDecodeError:
        return {"error": "Failed to parse LLM response into JSON.", "raw_response": response_str}
