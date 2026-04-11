import json
import logging
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
from llm import generate_response
from rag.retriever import retrieve_context

logger = logging.getLogger(__name__)

class DrugAnalysisResponse(BaseModel):
    uses: List[str] = Field(..., description="List of common uses for the drug")
    side_effects: List[str] = Field(..., description="List of potential side effects")
    dosage: str = Field(..., description="Typical dosage instructions")
    warnings: List[str] = Field(..., description="Safety warnings and precautions")
    drug_interactions: List[str] = Field(..., description="Known interactions with other substances")
    alternatives: List[str] = Field(..., description="Alternative medications or treatments")

def analyze_drug(drug_input: str) -> dict:
    """
    Analyzes a drug name or description using RAG and LLM to provide structured output.
    """
    context = retrieve_context(drug_input)
    
    system_prompt = """You are a medical AI assistant. Your job is to analyze the provided drug name/description and return a structured JSON response.
Always include this exact disclaimer in the warnings: 'Consult a medical professional before use'.
Format to return EXACTLY a JSON object matching this schema:
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
        data = json.loads(response_str)
        # Validate with Pydantic
        validated_data = DrugAnalysisResponse(**data)
        return validated_data.model_dump()
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Failed to validate LLM response: {e}")
        return {
            "error": "Failed to parse or validate LLM response into the required medical format.",
            "raw_response": response_str[:200] + "..." if len(response_str) > 200 else response_str
        }
